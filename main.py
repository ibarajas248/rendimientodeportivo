import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector

# Validar si los secretos están configurados
required_secrets = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
for secret in required_secrets:
    if secret not in st.secrets:
        st.error(f"Falta la configuración de {secret} en Streamlit Secrets.")
        st.stop()

# Usar variables de entorno desde Streamlit Secrets
DB_HOST = st.secrets["DB_HOST"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_NAME = st.secrets["DB_NAME"]

# Conectar a la base de datos usando las variables
def connect_to_database():
    try:
        return mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4'
        )
    except mysql.connector.Error as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return None

# Ejecutar consultas SQL
def run_query(query):
    try:
        db = connect_to_database()
        if db is None:
            return pd.DataFrame()
        cursor = db.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        db.close()
        return pd.DataFrame(result, columns=columns)
    except mysql.connector.Error as e:
        st.error(f"Error ejecutando la consulta SQL: {e}")
        return pd.DataFrame()

# Cargar lista de equipos
query_equipos = "SELECT id_equipo, equipo FROM equipo;"
df_equipos = run_query(query_equipos)

# Verificar si hay datos de equipos
if df_equipos.empty:
    st.error("No se encontraron equipos en la base de datos.")
else:
    st.sidebar.header("Seleccione Equipos")

    # Crear un filtro de selección de equipos
    equipos_display = df_equipos.set_index('id_equipo')['equipo'].to_dict()
    selected_ids = st.sidebar.multiselect(
        "Seleccione equipos",
        options=equipos_display.keys(),
        format_func=lambda x: equipos_display[x]
    )

    if not selected_ids:
        st.warning("Seleccione al menos un equipo para cargar datos.")
    else:
        # Crear consulta SQL con filtro por equipos
        equipos_placeholder = ','.join(map(str, selected_ids))  # Convierte IDs a formato SQL
        query_elo_filtrado = f"""
        SELECT 
            v.Equipo,
            v.id_equipo,
            v.elo_despues,
            v.id_partido,
            Partidos.id_torneo,
            v.elo_oponente
        FROM 
            (
                SELECT 
                    Equipo_local AS Equipo,
                    id_local AS id_equipo,
                    elo_local_despues AS elo_despues,
                    id_partido,
                    elo_visitante_despues AS elo_oponente
                FROM 
                    vista_elo 
                WHERE id_local IN ({equipos_placeholder})

                UNION ALL

                SELECT 
                    Equipo_visitante AS Equipo,
                    id_visitante AS id_equipo,
                    elo_visitante_despues AS elo_despues,
                    id_partido,
                    elo_local_despues AS elo_oponente
                FROM 
                    vista_elo 
                WHERE id_visitante IN ({equipos_placeholder})
            ) v
        INNER JOIN 
            Partidos
        ON 
            Partidos.id_partido = v.id_partido
        ORDER BY 
            v.id_partido DESC;
        """

        # Cargar datos filtrados
        dfEvolucionElo = run_query(query_elo_filtrado)

        if dfEvolucionElo.empty:
            st.warning("No se encontraron datos para los equipos seleccionados.")
        else:
            # Agregar control deslizante para limitar partidos
            max_partidos_slider = st.sidebar.slider(
                "Cantidad máxima de partidos",
                min_value=1,
                max_value=len(dfEvolucionElo),
                value=min(50, len(dfEvolucionElo))  # Valor inicial
            )

            # Aplicar límite al número de partidos
            filtered_data = dfEvolucionElo.head(max_partidos_slider)
            filtered_data['id_partido_aux'] = len(filtered_data) - np.arange(len(filtered_data))

            # Crear gráfico de evolución de ELO
            st.subheader("Evolución del ELO para Equipos Seleccionados")
            plt.figure(figsize=(10, 8))
            palette = sns.color_palette('tab10', n_colors=len(filtered_data['Equipo'].unique()))
            team_colors = dict(zip(filtered_data['Equipo'].unique(), palette))

            sns.lineplot(
                data=filtered_data,
                x='id_partido_aux',
                y='elo_despues',
                hue='Equipo',
                marker='o',
                palette=equipo_colores,
                linestyle='-'
            )

            # hace las x

            for equipo in filtered_data['Equipo'].unique():
                subset = filtered_data[filtered_data['Equipo'] == equipo]
                plt.scatter(
                    subset['id_partido_aux'],
                    subset['elo_oponente'],
                    marker='x',
                    color=equipo_colores[equipo],
                    label=f'ELO Oponente ({equipo})'
                )

            plt.xlabel('ID Partido Auxiliar (más antiguos a la izquierda)')
            plt.ylabel('ELO')
            plt.title('Evolución del ELO para Equipos Seleccionados')
            plt.legend(title='Equipo', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            st.pyplot(plt)

            resultado_colores = {
                'gano': 'green',  # Verde para ganados
                'perdio': 'red',  # Rojo para perdidos
                'empato': 'blue'  # Azul para empates
            }

            # Configurar la figura
            plt.figure(figsize=(10, 8))

            # Dibujar las líneas y puntos con diferentes colores según el resultado
            for equipo in filtered_data['Equipo'].unique():
                equipo_data = filtered_data[filtered_data['Equipo'] == equipo]

                # Dibujar líneas continuas para cada equipo
                plt.plot(
                    equipo_data['id_partido_aux'],
                    equipo_data['elo_despues'],
                    label=f"{equipo}",  # Etiqueta del equipo
                    color=equipo_colores[equipo],  # Usar el color asignado al equipo
                    linestyle='-', linewidth=2
                )

                # Dibujar los puntos con colores específicos según el resultado
                for resultado, color in resultado_colores.items():
                    subset = equipo_data[equipo_data['resultado'] == resultado]
                    plt.scatter(
                        subset['id_partido_aux'],  # Eje X
                        subset['elo_despues'],  # Eje Y
                        color=color,  # Usar el color según el resultado
                        marker='o',  # Marcador fijo (puedes cambiar esto si lo deseas)

                        label=f"{resultado.capitalize()} ({equipo})" if resultado in equipo_data[
                            'resultado'].unique() else None
                    )
                for equipo in filtered_data['Equipo'].unique():
                    subset = filtered_data[filtered_data['Equipo'] == equipo]
                    plt.scatter(
                        subset['id_partido_aux'],
                        subset['elo_oponente'],
                        marker='x',
                        color=equipo_colores[equipo],
                        label=f'ELO Oponente ({equipo})'
                    )

            # Etiquetas y título
            plt.xlabel('ID Partido Auxiliar (más antiguos a la izquierda)')
            plt.ylabel('ELO')
            plt.title('Evolución del ELO para Equipos Seleccionados')
            plt.legend(title='Equipo y Resultado', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True)

            # Mostrar la gráfica en Streamlit
            st.pyplot(plt)

            resultado_colores = {
                'gano': 'green',  # Verde para ganados
                'perdio': 'red',  # Rojo para perdidos
                'empato': 'blue'  # Azul para empates
            }

            # Configurar la figura
            plt.figure(figsize=(10, 8))

            # Dibujar las líneas y puntos con diferentes colores según el resultado
            for equipo in filtered_data['Equipo'].unique():
                equipo_data = filtered_data[filtered_data['Equipo'] == equipo]

                # Dibujar líneas continuas para cada equipo
                plt.plot(
                    equipo_data['id_partido_aux'],
                    equipo_data['elo_despues'],
                    label=f"{equipo}",  # Etiqueta del equipo
                    color=equipo_colores[equipo],  # Usar el color asignado al equipo
                    linestyle='-', linewidth=2
                )

                # Dibujar los puntos con colores específicos según el resultado
                for resultado, color in resultado_colores.items():
                    subset = equipo_data[equipo_data['resultado'] == resultado]
                    plt.scatter(
                        subset['id_partido_aux'],  # Eje X
                        subset['elo_despues'],  # Eje Y
                        color=color,  # Usar el color según el resultado
                        marker='o',  # Marcador fijo (puedes cambiar esto si lo deseas)

                        label=f"{resultado.capitalize()} ({equipo})" if resultado in equipo_data[
                            'resultado'].unique() else None
                    )

            # Etiquetas y título
            plt.xlabel('ID Partido Auxiliar (más antiguos a la izquierda)')
            plt.ylabel('ELO')
            plt.title('Evolución del ELO para Equipos Seleccionados')
            plt.legend(title='Equipo y Resultado', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True)

            # Mostrar la gráfica en Streamlit
            st.pyplot(plt)

            # Mostrar datos filtrados
            st.write("Datos filtrados:")
            st.dataframe(filtered_data)

            df_limpio = filtered_data[
                ['Equipo', 'contrincante', 'goles_local', 'goles_equipo_visitante', 'resultado', 'posicion',
                 'id_equipo']]

            # Filtrar partidos donde ambos equipos (Equipo y contrincante) están en los seleccionados
            df_limpio_filtrado = df_limpio[
                (df_limpio['id_equipo'].isin(selected_ids)) &
                (df_limpio['contrincante'].isin(
                    df_equipos[df_equipos['id_equipo'].isin(selected_ids)]['equipo'].values))
                ]

            # Mostrar el DataFrame filtrado en Streamlit
            st.subheader("Partidos entre Equipos Seleccionados")
            st.dataframe(df_limpio_filtrado)
            # Contar la cantidad de partidos ganados, empatados y perdidos por equipo
            df_resultados = df_limpio_filtrado.groupby(['Equipo', 'resultado']).size().reset_index(name='cantidad')

            # Crear el gráfico de barras
            plt.figure(figsize=(10, 6))

            # Dibujar un gráfico de barras para cada resultado
            sns.barplot(data=df_resultados, x='Equipo', y='cantidad', hue='resultado')

            # Configurar etiquetas y título
            plt.xlabel('Equipo')
            plt.ylabel('Cantidad de Partidos')
            plt.title('Resultados de Partidos por Equipo (Ganados, Empatados y Perdidos)')
            plt.legend(title='Resultado')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Mostrar el gráfico en Streamlit
            st.subheader("Resultados de Partidos por Equipo")
            st.pyplot(plt)