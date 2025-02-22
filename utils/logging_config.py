import logging

# Configurar el logger
logging.basicConfig(
    filename="static/app_logs.log",  # Nombre del archivo de logs
    level=logging.INFO,  # Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Formato del log
    datefmt="%Y-%m-%d %H:%M:%S",  # Formato de fecha
)

# Crear un logger
logger = logging.getLogger(__name__)
