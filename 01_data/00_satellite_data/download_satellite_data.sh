#!/bin/bash

# Incluir el archivo de configuración de tu shell para asegurarte de que el alias esté disponible
# Esto es comúnmente ~/.bashrc, ~/.bash_profile, o ~/.profile dependiendo de tu configuración
source ~/.zshrc  # o ~/.bash_profile o ~/.profile

# Archivo TSV con las fechas únicas
unique_dates_file="unique_dates_yyyymmdd.tsv"

# Directorio remoto y usuario
remote_user="ntoro"
remote_host="leftraru.nlhpc.cl"
remote_dir="kopas/satellite_data"

# Directorio local donde se guardarán los archivos descargados
local_dir="."

# Leer cada línea del archivo TSV, omitiendo el encabezado
tail -n +2 "$unique_dates_file" | while IFS= read -r date; do
  echo "Procesando fecha: $date"
  # Realizar la transferencia con scp
  scp -P 4603 "${remote_user}@${remote_host}:${remote_dir}/*${date}*.9km.nc" "$local_dir"
done

echo "Download completed."
