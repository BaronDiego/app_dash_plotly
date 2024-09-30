import pandas as pd
import json


df = pd.DataFrame({
    "Fecha":['22/01/2024','23/01/2024','24/01/2024','25/01/2024','26/01/2024','27/01/2024','29/01/2024','30/01/2024','31/01/2024','01/02/2024','02/02/2024','03/02/2024',
             '05/02/2024','06/02/2024','07/02/2024','08/02/2024','09/02/2024','10/02/2024','12/02/2024','13/02/2024','14/02/2024','15/02/2024','16/02/2024','17/02/2024',
             '19/02/2024','20/02/2024','21/02/2024','22/02/2024','23/02/2024','24/02/2024','26/02/2024','27/02/2024','28/02/2024','29/02/2024','01/03/2024','02/03/2024',
             '04/03/2024','05/03/2024','06/03/2024','07/03/2024','08/03/2024','09/03/2024','11/03/2024','12/03/2024','13/03/2024','14/03/2024','15/03/2024','16/03/2024',
             '18/03/2024','19/03/2024','20/03/2024','21/03/2024','22/03/2024','23/03/2024','25/03/2024','26/03/2024','27/03/2024','28/03/2024','29/03/2024','30/03/2024',
             '01/04/2024','02/04/2024','03/04/2024','04/04/2024','05/04/2024','06/04/2024','08/04/2024','09/04/2024','10/04/2024','11/04/2024','12/04/2024','13/04/2024',
             '15/04/2024','16/04/2024','17/04/2024','18/04/2024','19/04/2024','20/04/2024','22/04/2024','23/04/2024','24/04/2024','25/04/2024','26/04/2024','27/04/2024',
             '29/04/2024','30/04/2024','01/05/2024','02/05/2024','03/05/2024','04/05/2024','06/05/2024','07/05/2024','08/05/2024','09/05/2024','10/05/2024','11/05/2024',
],
    "Programado":[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                  0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.005,0.005,0.005,
                  0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.006,0.006,0.006,0.006,0.006,0.006,0.006,0.006,0.006,0.006,0.006,0.006,0.006,0.006,0.007,0.007,0.007,
                  0.007,0.007,0.007,0.007,0.007,0.007,0.008,0.009,0.010,0.010,0.010,0.010,0.010,0.010
],
    "Ejecutado":[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
                 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.003,0.003,0.003,
                 0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,
                 0.004,0.004,0.004,0.004,0.004,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005
]
})

df["Fecha"] = pd.to_datetime(df["Fecha"])


filename_in='data.json'
with open(filename_in, "r", encoding="utf8") as json_file:
    data = json.load(json_file)

datos = pd.DataFrame(data)
datos["Fecha"] = pd.to_datetime(datos["Fecha"])

print(datos)