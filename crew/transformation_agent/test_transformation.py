import os
import pandas as pd
from dotenv import load_dotenv

# Importación directa por estar en la misma carpeta
from transformation_agent import execute_transformation_with_sandbox

# 1. Cargar la API Key desde el archivo .env (si existe en la raíz)
load_dotenv()

# Si no usas .env, puedes desmarcar la siguiente línea y colocar tu clave directamente:
# os.environ["GOOGLE_API_KEY"] = "TU_API_KEY_DE_GOOGLE_STUDIO_AQUI"

# 2. Dataset de prueba con errores intencionales
data = {
    "Cliente": ["Juan", "Maria", None, "Carlos"],
    "Monto_Venta": ["$1200", "$500", "$3000", "invalido"],
    "Fecha": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
}
df_test = pd.DataFrame(data)

print("=== 1. DATAFRAME ORIGINAL ===")
print(df_test)
print("\n" + "=" * 40 + "\n")

# 3. Instrucción en lenguaje natural
prompt_prueba = "Elimina filas con Cliente nulo, limpia los símbolos de $ en Monto_Venta y convierte la columna a número eliminando valores no válidos."

print(f"📝 Prompt enviado: '{prompt_prueba}'\n")

# 4. Ejecutar el Agente y su Sandbox
try:
    df_limpio, script_generado = execute_transformation_with_sandbox(
        df_test, prompt_prueba
    )

    print("=== 2. DATAFRAME TRANSFORMADO ===")
    print(df_limpio)
    print("\n" + "=" * 40 + "\n")

    print("=== 3. CÓDIGO PYTHON GENERADO Y VALIDADO ===")
    print(script_generado)

except Exception as e:
    print(f"❌ Error durante la prueba: {e}")