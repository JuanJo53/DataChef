import os
import re
import sys
import time
import asyncio
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from google import genai

# Fix para conflictos de asyncio/sockets de Google API en Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Carga automáticamente el archivo .env buscando desde la raíz del proyecto
load_dotenv(find_dotenv())

# A "lite" ALIAS, not a fixed version. Two reasons:
#
# 1. ALIAS, not a version: Google retires old versions. gemini-2.5-flash and
#    gemini-2.5-flash-lite already return 404. The alias always tracks the
#    current model, so that error cannot come back.
# 2. LITE, not regular flash: on the free tier the quota is PER MODEL per day.
#    gemini-flash-latest points at the newest flash (today gemini-3.7-flash),
#    which carries the tightest quota: 20 requests/day. That runs out fast,
#    since one transformation can spend several (each Self-Healing is another
#    call). The lite family is the cheapest and has the most headroom.
MODEL = "gemini-flash-lite-latest"

# Retries ONLY for 503 (server busy). See _is_transient_error.
API_RETRIES = 3
RETRY_BASE_SECONDS = 4


def get_client():
    """Inicializa el cliente oficial de Google GenAI usando la API Key."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Falta la variable de entorno GOOGLE_API_KEY. Configúrala en tu archivo .env"
        )
    return genai.Client(api_key=api_key)


def _is_transient_error(e: Exception) -> bool:
    """True if the failure is "Google is busy" (503), which retrying DOES fix.

    Same classification smoke_test_gemini.py uses: key (401/403), model (404)
    and quota (429) errors do not improve by waiting, so those propagate
    immediately instead of making the user sit through a backoff.
    """
    msg = str(e)
    return "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg


def _generate_with_retries(client, contents: str) -> str:
    """Call Gemini, retrying ONLY when the server is busy.

    Without this a transient 503 (very common on the free tier) killed the
    whole transformation with a red error, even though retrying a few seconds
    later works. The wait grows on each attempt (linear backoff).
    """
    for attempt in range(1, API_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
            )
            return response.text
        except Exception as e:
            # Last attempt, or an error that waiting will not fix -> propagate.
            if attempt == API_RETRIES or not _is_transient_error(e):
                raise
            wait = RETRY_BASE_SECONDS * attempt
            print(
                f"[wait] Google servers are busy (attempt {attempt}/"
                f"{API_RETRIES}), retrying in {wait}s..."
            )
            time.sleep(wait)
    # Unreachable: the loop exits via return or raise.
    raise RuntimeError("retries exhausted")


def clean_code_block(text: str) -> str:
    """Extrae únicamente el bloque de código Python de la respuesta del LLM."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.replace("```python", "").replace("```", "").strip()


def generate_transformation_code(
    columns_info: str, user_prompt: str, client
) -> str:
    """Ask the LLM to generate the pandas script.

    The prompt is in English on purpose: the model answers in the language it
    is asked in, and a Spanish prompt produced Spanish comments inside the
    generated script that the user then sees on screen.
    """
    system_prompt = f"""
    You are a Senior Data Engineer, expert in Python and Pandas.

    A pandas DataFrame is loaded in the variable 'df' with these columns and dtypes:
    {columns_info}

    The user requested the following transformation in natural language:
    "{user_prompt}"

    STRICT OUTPUT RULES:
    1. Return ONLY executable Python code inside a ```python ... ``` block.
    2. Assume the DataFrame already exists in memory under the name `df`.
    3. Make sure the final processed result stays in the variable `df`.
    4. Write any code comments in English.
    5. If the user mentions a column that does not exist, do NOT invent it:
       apply the closest sensible operation on the real columns listed above.
    6. Do NOT add explanations or introductions outside the code block.
    """

    return clean_code_block(_generate_with_retries(client, system_prompt))


def fix_failing_code(
    failing_code: str, error_msg: str, columns_info: str, client
) -> str:
    """Self-healing: fix the pandas code that failed at execution time."""
    fix_prompt = f"""
    The following pandas code failed during execution:
    ```python
    {failing_code}
    ```

    It produced this Python error:
    "{error_msg}"

    Columns available on the DataFrame 'df':
    {columns_info}

    Analyse the error, fix it, and return ONLY the corrected code inside a
    ```python ... ``` block. Write any code comments in English.
    """

    return clean_code_block(_generate_with_retries(client, fix_prompt))


def execute_transformation_with_sandbox(
    df_original: pd.DataFrame, user_prompt: str, max_retries: int = 3
):
    """
    Función principal expuesta para el equipo (Streamlit/Backend).
    Ejecuta el código generado en un Local Sandbox y aplica Self-Healing si falla.
    """
    client = get_client()

    # Trabajar con una copia para evitar modificar el dataframe original si falla
    df_working = df_original.copy()
    columns_info = str(df_working.dtypes)

    print("[agent] Generating transformation script...")
    current_code = generate_transformation_code(
        columns_info, user_prompt, client
    )

    # Bucle del Local Sandbox con auto-corrección
    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[sandbox] [Attempt {attempt}/{max_retries}] Executing in Local Sandbox..."
            )

            # Creación del entorno aislado (Scope)
            sandbox_scope = {"pd": pd, "df": df_working}

            # Ejecución en Sandbox
            exec(current_code, sandbox_scope)

            # Recuperar el DataFrame transformado
            df_result = sandbox_scope["df"]
            print("[ok] Transformation executed successfully!")

            return df_result, current_code

        except Exception as e:
            error_message = str(e)
            print(f"[warn] Error on attempt {attempt}: {error_message}")

            if attempt < max_retries:
                print("[heal] Applying Self-Healing to repair the code...")
                current_code = fix_failing_code(
                    current_code, error_message, columns_info, client
                )

    raise RuntimeError(
        f"Error: transformation could not be executed after {max_retries} attempts."
    )


# =====================================================================
# FUNCIÓN PARA GUARDAR EL PIPELINE REUTILIZABLE
# =====================================================================
def save_pipeline_script(
    code_script: str, output_path: str = "pipeline_transformacion.py"
):
    """Guarda el código generado por el agente como un archivo .py reutilizable."""

    header = """# ========================================================
# PIPELINE DE TRANSFORMACIÓN AUTOMÁTICO - DATACHEF
# Este archivo fue generado automáticamente por Action Agent.
# Puedes ejecutarlo directamente sobre nuevos archivos CSV.
# ========================================================
import pandas as pd

def run_pipeline(input_csv_path: str, output_csv_path: str):
    # 1. Cargar datos crudos
    print(f"[load] Loading file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # 2. Aplicar Transformaciones Registradas
"""

    indented_lines = []
    for line in code_script.split("\n"):
        if line.strip():
            indented_lines.append("    " + line)
        else:
            indented_lines.append("")

    indented_code = "\n".join(indented_lines)

    footer = """

    # 3. Guardar resultado transformado
    df.to_csv(output_csv_path, index=False)
    print(f"[ok] Pipeline executed successfully. Saved to: {output_csv_path}")
    return df

if __name__ == "__main__":
    run_pipeline("nuevos_datos_crudos.csv", "datos_limpios_resultado.csv")
"""

    full_script = header + indented_code + footer

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_script)

    print(f"[saved] Reusable pipeline saved to '{output_path}'!")
    return output_path