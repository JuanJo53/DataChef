import os
import pandas as pd
from dotenv import load_dotenv

# Importar las dos funciones desde tu agente
from transformation_agent import (
    execute_transformation_with_sandbox,
    save_pipeline_script,
)

# 1. Cargar la API Key desde el archivo .env
load_dotenv()

# 2. Dataset de prueba con datos "sucios"
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
    print("\n" + "=" * 40 + "\n")

    # 🚀 5. Probar la generación del Pipeline Reutilizable (.py)
    archivo_pipeline = save_pipeline_script(
        script_generado, "pipeline_prueba.py"
    )
    print(f"🎉 ¡Pipeline guardado exitosamente como '{archivo_pipeline}'!")

except Exception as e:
    print(f"❌ Error durante la prueba: {e}")