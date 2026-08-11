import os
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    """Inicializa el modelo Gemini 1.5 Pro usando la API Key de las variables de entorno."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Falta la variable de entorno GOOGLE_API_KEY. Configúrala en tu archivo .env"
        )

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,  # Temperatura baja para garantizar código de Python sintácticamente correcto
        google_api_key=api_key,
    )


def clean_code_block(text: str) -> str:
    """Extrae únicamente el bloque de código Python de la respuesta del LLM."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.replace("```python", "").replace("```", "").strip()


def generate_transformation_code(
    columns_info: str, user_prompt: str, llm
) -> str:
    """Solicita al LLM la generación del script de Pandas."""
    system_prompt = f"""
    Eres un Senior Data Engineer experto en Python y Pandas.
    
    Tienes un DataFrame de Pandas cargado en la variable 'df' con las siguientes columnas y tipos de datos:
    {columns_info}

    El usuario solicitó la siguiente transformación en lenguaje natural:
    "{user_prompt}"

    REGLAS ESTRICTAS DE SALIDA:
    1. Devuelve ÚNICAMENTE código ejecutable de Python dentro de un bloque ```python ... ```.
    2. Asume que el DataFrame ya existe en memoria bajo el nombre `df`.
    3. Asegúrate de que el resultado final procesado se mantenga/guarde en la variable `df`.
    4. NO agregues explicaciones, comentarios de texto ni introducción. Solo el código Python.
    """

    response = llm.invoke(system_prompt)
    return clean_code_block(response.content)


def fix_failing_code(
    failing_code: str, error_msg: str, columns_info: str, llm
) -> str:
    """Auto-corrige el código de Python que falló en la ejecución (Self-Healing)."""
    fix_prompt = f"""
    El siguiente código de Pandas falló durante la ejecución:
    ```python
    {failing_code}
    ```

    Produjo el siguiente error de Python:
    "{error_msg}"

    Información de columnas del DataFrame 'df':
    {columns_info}

    Por favor, analiza el error, corrígelo y devuelve ÚNICAMENTE el código corregido dentro de un bloque ```python ... ```.
    """

    response = llm.invoke(fix_prompt)
    return clean_code_block(response.content)


def execute_transformation_with_sandbox(
    df_original: pd.DataFrame, user_prompt: str, max_retries: int = 3
):
    """
    Función principal expuesta para el equipo (Streamlit/Backend).
    Ejecuta el código generado en un Local Sandbox y aplica Self-Healing si falla.
    """
    llm = get_llm()

    # Trabajar con una copia para evitar modificar el dataframe original si falla
    df_working = df_original.copy()
    columns_info = str(df_working.dtypes)

    print("🤖 Generando script de transformación...")
    current_code = generate_transformation_code(
        columns_info, user_prompt, llm
    )

    # Bucle del Local Sandbox con auto-corrección
    for attempt in range(1, max_retries + 1):
        try:
            print(f"⚙️ [Intento {attempt}/{max_retries}] Ejecutando en Local Sandbox...")

            # Creación del entorno aislado (Scope)
            sandbox_scope = {"pd": pd, "df": df_working}

            # Ejecución en Sandbox
            exec(current_code, sandbox_scope)

            # Recuperar el DataFrame transformado
            df_result = sandbox_scope["df"]
            print("✅ ¡Transformación ejecutada exitosamente!")

            return df_result, current_code

        except Exception as e:
            error_message = str(e)
            print(f"⚠️ Error en intento {attempt}: {error_message}")

            if attempt < max_retries:
                print("🔄 Aplicando Self-Healing para reparar el código...")
                current_code = fix_failing_code(
                    current_code, error_message, columns_info, llm
                )

    raise RuntimeError(
        f"Error: No se pudo ejecutar la transformación tras {max_retries} intentos."
    )


# =====================================================================
# 📄 FUNCIÓN PARA GUARDAR EL PIPELINE REUTILIZABLE
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
    print(f"📂 Cargando archivo: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # 2. Aplicar Transformaciones Registradas
"""

    # Indentar correctamente cada línea del script para que encaje dentro de la función run_pipeline
    indented_lines = []
    for line in code_script.split("\n"):
        if line.strip():  # Si la línea tiene contenido
            indented_lines.append("    " + line)
        else:  # Si es una línea en blanco
            indented_lines.append("")

    indented_code = "\n".join(indented_lines)

    footer = """

    # 3. Guardar resultado transformado
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Pipeline ejecutado con éxito. Guardado en: {output_csv_path}")
    return df

if __name__ == "__main__":
    # Cambia los nombres de los archivos según tu necesidad
    run_pipeline("nuevos_datos_crudos.csv", "datos_limpios_resultado.csv")
"""

    full_script = header + indented_code + footer

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_script)

    print(f"📄 ¡Pipeline reutilizable guardado con éxito en '{output_path}'!")
    return output_path