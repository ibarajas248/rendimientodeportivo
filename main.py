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
            v.contrincante,
            v.id_equipo,
            v.elo_despues,
            v.id_partido,
            Partidos.id_torneo,
            v.elo_oponente,
            Partidos.goles_local,
            Partidos.goles_equipo_visitante,
            v.posicion,
            CASE 
                WHEN v.posicion = 'local' AND Partidos.goles_local > Partidos.goles_equipo_visitante THEN 'gano'
                WHEN v.posicion = 'local' AND Partidos.goles_local < Partidos.goles_equipo_visitante THEN 'perdio'
                WHEN v.posicion = 'local' AND Partidos.goles_local = Partidos.goles_equipo_visitante THEN 'empato'
                WHEN v.posicion = 'visitante' AND Partidos.goles_equipo_visitante > Partidos.goles_local THEN 'gano'
                WHEN v.posicion = 'visitante' AND Partidos.goles_equipo_visitante < Partidos.goles_local THEN 'perdio'
                WHEN v.posicion = 'visitante' AND Partidos.goles_equipo_visitante = Partidos.goles_local THEN 'empato'
            END AS resultado

        FROM 
            (
                SELECT 
                    Equipo_local AS Equipo,
                    Equipo_visitante AS contrincante,
                    id_local AS id_equipo,
                    elo_local_despues AS elo_despues,
                    id_partido,
                    elo_visitante_despues AS elo_oponente,
                    'local' AS posicion

                FROM 
                    vista_elo 
                WHERE id_local IN ({equipos_placeholder})

                UNION ALL

                SELECT 
                    Equipo_visitante AS Equipo,
                     Equipo_local AS contrincante,
                    id_visitante AS id_equipo,
                    elo_visitante_despues AS elo_despues,
                    id_partido,
                    elo_local_despues AS elo_oponente,
                    'visitante' AS posicion
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

            # Selección de colores para cada equipo
            st.sidebar.subheader("Colores de las líneas")
            equipo_colores = {}
            default_colors = ['#FF0000', '#0000FF']  # Rojo para el primero, azul para el segundo

            # Obtener equipos únicos
            equipos_unicos = list(set(filtered_data['Equipo']))

            # Asignar colores
            for idx, equipo in enumerate(equipos_unicos):
                # Asignar color predeterminado para los dos primeros equipos
                if idx < len(default_colors):
                    default_color = default_colors[idx]
                else:
                    # Para los equipos restantes, asignar un color blanco por defecto
                    default_color = '#FFFFFF'

                # Muestra el selector de colores con el color predeterminado
                color = st.sidebar.color_picker(f"Color para {equipo}", value=default_color)

                # Guarda el color seleccionado
                equipo_colores[equipo] = color

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

import pyttsx3
import tempfile
import os
import streamlit as st
import openai


def obtener_analisis(data, max_tokens=500):
    try:
        # El contenido que vamos a enviar a OpenAI para el análisis
        mensaje = f"""
        Aquí está el gráfico de la evolución del ELO para los equipos seleccionados junto con los datos:

        Datos:
        {data}

        Haz un análisis de los datos, al final dame una probabilidad de victoria por equipo, en porcentaje. Ejemplo: equipo A 67% de probabilidad frente a equipo B en el próximo partido y hay un 20 % probabilidad de empate. considera especialmente los ultimas tendencias
        si se puede estima tambien probabilidad , al final debe decir equipo A 67% de probabilidad frente a equipo B
        """

        # Enviar solicitud a OpenAI para obtener el análisis usando el modelo de chat
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Asegúrate de usar el modelo correcto
            messages=[
                {"role": "system", "content": "Eres un analista de datos deportivo."},
                {"role": "user", "content": mensaje}
            ],
            max_tokens=max_tokens,  # Limitar el número de tokens generados en la respuesta
            temperature=0.7  # Ajusta la creatividad (0 = más predecible, 1 = más creativo)
        )

        analysis_text = response['choices'][0]['message']['content'].strip()

        # Ejemplo de cómo podrías incluir las fórmulas de probabilidad en formato LaTeX
        formula = r"""
        P(A) = \frac{1}{1 + 10^{\frac{(ELO_B - ELO_A)}{400}}}
        """

        # Mostrar el análisis en Streamlit
        st.subheader("Análisis de rendimiento y probabilidades de victoria")
        st.write(analysis_text)

        # Mostrar la fórmula en LaTeX en Streamlit
        st.latex(formula)

        return analysis_text

    except Exception as e:
        return f"Error en la consulta a OpenAI: {str(e)}"


# Suponiendo que tienes los datos de entrada en la variable 'filtered_data'
if st.button('Analizar rendimiento'):
    # Obtener análisis de OpenAI basado en los datos generados
    analysis_result = obtener_analisis(filtered_data)

    # Mostrar el resultado en Streamlit
    st.subheader("Análisis del Rendimiento")
    st.write(analysis_result)