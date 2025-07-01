import wandb
import yaml  # Para guardar como archivo YAML legible

# Define tus datos
entity = "citius-irlab"
project = "early_attention_reseatch"
sweep_id = "433xougl"  # solo el ID final, no la ruta completa

# Autenticación (si no estás logueado ya)
wandb.login()

# Accede al sweep
api = wandb.Api()
sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

# Obtén la configuración
sweep_config = sweep.config

# Imprime o guarda como YAML
with open("sweep_config.yaml", "w") as f:
    yaml.dump(sweep_config, f)

print("Configuración guardada en sweep_config.yaml")
