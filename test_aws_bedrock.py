"""
Test rápido de conectividad con AWS Bedrock.
Ejecutar: python test_aws_bedrock.py
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(".env")

print("=" * 55)
print("  DIAGNÓSTICO AWS BEDROCK")
print("=" * 55)

# 1. Verificar variables de entorno
print("\n[1/3] Verificando credenciales en .env...")
key = os.getenv("AWS_ACCESS_KEY_ID", "")
secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
region = os.getenv("AWS_REGION", "us-east-2")

if not key or not secret:
    print("  ❌ FALTA: AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY en .env")
    sys.exit(1)

print(f"  ✅ Access Key: {key[:8]}...{key[-4:]}")
print(f"  ✅ Secret Key: {'*' * 20} (presente)")
print(f"  ✅ Region: {region}")

# 2. Probar boto3 client
print("\n[2/3] Probando conexión boto3 a Bedrock Runtime...")
try:
    import boto3
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )
    print("  ✅ Cliente boto3 creado exitosamente")
except Exception as e:
    print(f"  ❌ Error creando cliente boto3: {e}")
    sys.exit(1)

# 3. Probar embedding real con timeout visual
print("\n[3/3] Probando llamada a Titan Embed v2 (timeout: 30s)...")
import json

try:
    t0 = time.time()
    body = json.dumps({"inputText": "Tutela de salud Colombia"})
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    elapsed = time.time() - t0
    result = json.loads(response["body"].read())
    dims = len(result.get("embedding", []))
    print(f"  ✅ Embedding generado en {elapsed:.2f}s")
    print(f"  ✅ Dimensiones del vector: {dims}")
    print(f"\n{'=' * 55}")
    print("  ✅ AWS BEDROCK FUNCIONA CORRECTAMENTE")
    print(f"  → Puedes proceder con la ingesta de PDFs")
    print(f"{'=' * 55}")
except client.exceptions.AccessDeniedException:
    print("  ❌ AccessDeniedException: El modelo NO está habilitado en Bedrock")
    print("     → Ve a AWS Console > Bedrock > Model Access")
    print("     → Habilita: Amazon Titan Embed Text v2")
except Exception as e:
    print(f"  ❌ Error en llamada a Bedrock: {type(e).__name__}: {e}")
    print("  → Verifica acceso a Bedrock en la consola AWS")
