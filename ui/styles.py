import streamlit as st

def _apply_custom_styles():
    st.markdown(
        """
        <style>
        /* =========================================================
        DATACHEF FILE UPLOADER
        ========================================================= */

        [data-testid="stFileUploader"] {
            width: 100%;
        }


        /* Caja grande */
        [data-testid="stFileUploaderDropzone"] {
            position: relative !important;

            min-height: 300px !important;

            border: 2px dashed #8b5cf6 !important;
            border-radius: 18px !important;

            background:
                radial-gradient(
                    circle at center,
                    rgba(99, 102, 241, 0.07),
                    transparent 55%
                ) !important;

            padding: 130px 30px 35px 30px !important;

            display: flex !important;
            align-items: center !important;
            justify-content: center !important;

            transition: all 0.25s ease !important;
        }


        /* Hover */
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #3b82f6 !important;

            box-shadow:
                0 0 25px rgba(99, 102, 241, 0.10) !important;
        }


       /* =========================================================
        NUBE GRANDE
        ========================================================= */

        [data-testid="stFileUploaderDropzone"]::before {

            content: "";

            position: absolute;

            top: 32px;
            left: 50%;

            transform: translateX(-50%);

            /* Proporcional al tamaño de la caja */
            width: clamp(90px, 8vw, 135px);
            height: clamp(75px, 7vw, 115px);

            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;

            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='80' viewBox='0 0 24 24' fill='none' stroke='%235B7CFF' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M16 16l-4-4-4 4'/%3E%3Cpath d='M12 12v9'/%3E%3Cpath d='M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3'/%3E%3Cpath d='M16 16l-4-4-4 4'/%3E%3C/svg%3E");

            filter:
                drop-shadow(
                    0 0 12px
                    rgba(91, 124, 255, 0.50)
                );

            pointer-events: none;
        }

/* =========================================================
   ESTRUCTURA INTERNA DEL FILE UPLOADER
========================================================= */

[data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    flex-direction: column !important;

    align-items: center !important;
    justify-content: center !important;
}


/* Primer contenedor interno */
[data-testid="stFileUploaderDropzone"] > div {
    width: 100% !important;

    display: flex !important;
    flex-direction: column !important;

    align-items: center !important;
    justify-content: center !important;

    text-align: center !important;

    gap: 12px !important;
}


/* TODOS los wrappers internos */
[data-testid="stFileUploaderDropzone"] > div > div {
    width: 100% !important;

    display: flex !important;
    flex-direction: column !important;

    align-items: center !important;
    justify-content: center !important;

    text-align: center !important;

    margin-left: auto !important;
    margin-right: auto !important;
}


/* =========================================================
   INSTRUCCIONES
========================================================= */

[data-testid="stFileUploaderDropzoneInstructions"] {
    width: 100% !important;

    display: flex !important;
    flex-direction: column !important;

    align-items: center !important;
    justify-content: center !important;

    text-align: center !important;

    margin: 0 auto !important;
}


/* =========================================================
   BOTÓN
========================================================= */

[data-testid="stFileUploaderDropzone"] button {
    display: flex !important;

    align-items: center !important;
    justify-content: center !important;

    align-self: center !important;

    position: static !important;

    float: none !important;

    width: 160px !important;
    min-width: 160px !important;

    height: 44px !important;

    margin:
        18px auto 0 auto !important;

    padding:
        0 24px !important;

    border-radius:
        9px !important;

    border:
        1px solid #3b82f6 !important;

    background:
        transparent !important;

    color:
        #60a5fa !important;

    font-size:
        14px !important;

    font-weight:
        600 !important;
}


/* Wrapper específico que Streamlit usa alrededor del botón */
[data-testid="stFileUploaderDropzone"] button.parent,
[data-testid="stFileUploaderDropzone"] button-wrapper {
    width: 100% !important;

    display: flex !important;

    justify-content: center !important;

    align-items: center !important;
}

[data-testid="stFileUploaderDropzone"] > div {
    grid-template-columns: 1fr !important;
}

/* =========================================================
   HOVER - GLOW DEL BOTÓN UPLOAD
========================================================= */

[data-testid="stFileUploaderDropzone"] button {
    transition:
        background 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease,
        transform 0.25s ease !important;
}


[data-testid="stFileUploaderDropzone"] button:hover {

    /* Fondo azul muy transparente */
    background:
        rgba(59, 130, 246, 0.10) !important;

    /* Borde un poco más brillante */
    border-color:
        #60a5fa !important;

    /* Glow */
    box-shadow:
        0 0 6px rgba(96, 165, 250, 0.65),
        0 0 15px rgba(59, 130, 246, 0.40),
        0 0 28px rgba(99, 102, 241, 0.25) !important;

    /* Movimiento muy pequeño */
    transform:
        translateY(-2px) scale(1.02);

    color:
        #93c5fd !important;
}





        @import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;500;600;700&display=swap');







        /* =========================================================
        METRIC CARDS
        ========================================================= */

        /* Estilo base para TODAS las cards */
        [data-testid="stMetric"] {

            padding: 20px 22px;

            border-radius: 14px;

            min-height: 105px;

            backdrop-filter: blur(10px);

            box-shadow:
                0 8px 22px
                rgba(0, 0, 0, 0.08);

            transition:
                transform 0.2s ease,
                box-shadow 0.2s ease;
        }


        /* Hover */
        [data-testid="stMetric"]:hover {

            transform: translateY(-3px);

            box-shadow:
                0 12px 28px
                rgba(0, 0, 0, 0.16);
        }


        /* =========================================================
        CARD 1 - ROWS - GREEN
        ========================================================= */

        [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"]:nth-child(1)
        [data-testid="stMetric"] {

            background:
                rgba(16, 185, 129, 0.12);

            border:
                1px solid
                rgba(16, 185, 129, 0.30);
        }


        /* =========================================================
        CARD 2 - COLUMNS - BLUE
        ========================================================= */

        [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"]:nth-child(2)
        [data-testid="stMetric"] {

            background:
                rgba(59, 130, 246, 0.12);

            border:
                1px solid
                rgba(59, 130, 246, 0.30);
        }


        /* =========================================================
        CARD 3 - DATA TYPES - PURPLE
        ========================================================= */

        [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"]:nth-child(3)
        [data-testid="stMetric"] {

            background:
                rgba(168, 85, 247, 0.12);

            border:
                1px solid
                rgba(168, 85, 247, 0.30);
        }


        /* =========================================================
        LABEL
        ========================================================= */

        [data-testid="stMetricLabel"] {

            font-size: 14px !important;

            opacity: 0.75;
        }


        /* =========================================================
        VALUE
        ========================================================= */

        [data-testid="stMetricValue"] {

            font-size: 28px !important;

            font-weight: 700 !important;
        }


        /* =========================================================
        DATA PREVIEW - GLASS EFFECT
        ========================================================= */

        [data-testid="stDataFrame"] {

            /* Blanco transparente */
            background:
                rgba(255, 255, 255, 0.1) !important;

            /* Borde blanco suave */
            border:
                1px solid
                rgba(255, 255, 255, 0.14) !important;

            border-radius:
                14px !important;

            overflow:
                hidden !important;

            /* Efecto glass */
            backdrop-filter:
                blur(12px);

            -webkit-backdrop-filter:
                blur(12px);

            /* Sombra */
            box-shadow:
                0 8px 25px
                rgba(0, 0, 0, 0.12);
        }



        </style>
        """,
        unsafe_allow_html=True,
    )
