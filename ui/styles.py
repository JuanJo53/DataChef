import streamlit as st


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>

        /* ========================================
           SIDEBAR
        ======================================== */

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                rgba(18, 18, 32, 0.98) 0%,
                rgba(25, 20, 45, 0.98) 100%
            );
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }


        /* ========================================
           PROGRESS TITLE
        ======================================== */

        .progress-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #F4F4F5;

            margin-top: 1rem;
            margin-bottom: 0.8rem;

            letter-spacing: 0.3px;
        }


        /* ========================================
           CURRENT STAGE
        ======================================== */

        .stage-current {
            display: flex;
            align-items: center;
            gap: 10px;

            padding: 14px 16px;
            margin: 8px 0;

            border-radius: 14px;

            background: linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.30),
                rgba(99, 102, 241, 0.16)
            );

            border: 1px solid rgba(167, 139, 250, 0.50);

            box-shadow:
                0 5px 18px rgba(124, 58, 237, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);

            font-size: 1.08rem;
            font-weight: 700;

            color: #DDD6FE;

            transform: scale(1.025);
            transition: all 0.25s ease;
        }


        /* ========================================
           PENDING STAGE
        ======================================== */

        .stage-pending {
            display: flex;
            align-items: center;
            gap: 10px;

            padding: 10px 14px;
            margin: 5px 0;

            border-radius: 11px;

            color: rgba(226, 232, 240, 0.50);

            font-size: 0.92rem;
            font-weight: 500;
        }


        /* ========================================
           COMPLETED STAGE BUTTONS
        ======================================== */

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;

            padding: 0.55rem 0.8rem;

            border-radius: 11px;
            border: 1px solid rgba(255, 255, 255, 0.08);

            background: rgba(255, 255, 255, 0.035);

            color: rgba(226, 232, 240, 0.82);

            font-size: 0.92rem;
            font-weight: 500;

            transition: all 0.20s ease;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(124, 58, 237, 0.14);

            border-color: rgba(139, 92, 246, 0.35);

            color: #E9D5FF;

            transform: translateX(3px);
        }


        /* ========================================
           DIVIDERS
        ======================================== */

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.08);

            margin-top: 1.3rem;
            margin-bottom: 1.3rem;
        }


        /* ========================================
           SIDEBAR LOGO
        ======================================== */

        [data-testid="stSidebar"] [data-testid="stImage"] img {
            display: block;

            width: 100% !important;
            max-width: 260px !important;
            height: auto !important;

            margin-left: auto !important;
            margin-right: auto !important;

            object-fit: contain;
        }


        /* ========================================
           MAIN SUBTITLE
        ======================================== */

        .main-subtitle {
            text-align: center;

            font-family: "Century Gothic", "Segoe UI", sans-serif;

            font-size: 20px;
            font-weight: 500;

            letter-spacing: 1.8px;

            color: #FFFFFF;

            margin-top: -12px;
            margin-bottom: 32px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )



def upload_styles() -> None:
    st.markdown(
        """
        <style>
    /* ========================================
    FILE UPLOADER
    ======================================== */

    /* Contenedor general del uploader */
    [data-testid="stFileUploader"] {
        width: 100%;
    }


    /* Dropzone principal */
    [data-testid="stFileUploaderDropzone"] {
        min-height: 320px;

        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;

        position: relative;

        padding: 135px 30px 45px 30px !important;

        border: 2px dashed rgba(139, 92, 246, 0.75) !important;
        border-radius: 20px !important;

        background:
            radial-gradient(
                circle at center,
                rgba(99, 102, 241, 0.10) 0%,
                rgba(124, 58, 237, 0.05) 45%,
                rgba(15, 15, 30, 0.02) 100%
            ) !important;

        transition:
            border-color 0.3s ease,
            box-shadow 0.3s ease,
            background 0.3s ease,
            transform 0.3s ease;
    }


    /* Hover sobre toda la caja */
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(167, 139, 250, 1) !important;

        background:
            radial-gradient(
                circle at center,
                rgba(124, 58, 237, 0.14) 0%,
                rgba(99, 102, 241, 0.08) 45%,
                rgba(15, 15, 30, 0.02) 100%
            ) !important;

        box-shadow:
            0 0 20px rgba(124, 58, 237, 0.18),
            0 0 45px rgba(99, 102, 241, 0.10),
            inset 0 0 25px rgba(124, 58, 237, 0.04);

        transform: translateY(-2px);
    }


    /* ========================================
    CLOUD ICON
    ======================================== */

    [data-testid="stFileUploaderDropzone"]::before {
        content: "☁";

        position: absolute;

        top: 42px;
        left: 50%;
        transform: translateX(-50%);

        font-size: 72px;
        line-height: 1;

        color: rgba(167, 139, 250, 0.90);

        text-shadow:
            0 0 12px rgba(139, 92, 246, 0.50),
            0 0 30px rgba(99, 102, 241, 0.30);

        pointer-events: none;

        transition:
            transform 0.3s ease,
            color 0.3s ease,
            text-shadow 0.3s ease;
    }


    /* Animación de la nube */
    [data-testid="stFileUploaderDropzone"]:hover::before {
        transform: translateX(-50%) translateY(-4px) scale(1.07);

        color: #C4B5FD;

        text-shadow:
            0 0 18px rgba(167, 139, 250, 0.75),
            0 0 35px rgba(124, 58, 237, 0.45);
    }


    /* ========================================
    UPLOAD TEXT
    ======================================== */

    [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] {
        text-align: center;
    }


    /* Texto de Drag and drop */
    [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
        color: rgba(226, 232, 240, 0.82) !important;

        font-size: 1rem !important;
        font-weight: 500;

        text-align: center;
    }


    /* Información de límite/formato */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        text-align: center;
    }


    /* ========================================
    BROWSE / UPLOAD BUTTON
    ======================================== */

    [data-testid="stFileUploaderDropzone"] button {
        border-radius: 12px !important;

        padding: 0.6rem 1.5rem !important;

        border: 1px solid rgba(139, 92, 246, 0.65) !important;

        background: linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.75),
            rgba(99, 102, 241, 0.65)
        ) !important;

        color: #FFFFFF !important;

        font-weight: 600 !important;

        box-shadow:
            0 4px 15px rgba(124, 58, 237, 0.20);

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease,
            border-color 0.25s ease,
            filter 0.25s ease;
    }


    /* Glow del botón */
    [data-testid="stFileUploaderDropzone"] button:hover {
        transform: translateY(-2px) scale(1.03);

        border-color: rgba(196, 181, 253, 0.95) !important;

        filter: brightness(1.12);

        box-shadow:
            0 0 12px rgba(139, 92, 246, 0.65),
            0 0 25px rgba(124, 58, 237, 0.40),
            0 7px 20px rgba(99, 102, 241, 0.25);
    }


    /* Click */
    [data-testid="stFileUploaderDropzone"] button:active {
        transform: scale(0.98);
    }


    /* ========================================
   DATASET METRIC CARDS
======================================== */

.metric-card {
    position: relative;

    min-height: 150px;

    display: flex;
    align-items: center;

    gap: 24px;

    padding: 28px 30px;

    border-radius: 22px;

    overflow: hidden;

    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        border-color 0.25s ease;
}


/* pequeño efecto luminoso interno */
.metric-card::after {
    content: "";

    position: absolute;

    width: 220px;
    height: 130px;

    right: -30px;
    bottom: -65px;

    border-radius: 50%;

    background: rgba(255, 255, 255, 0.035);

    filter: blur(4px);

    pointer-events: none;
}


/* ========================================
   PURPLE CARD
======================================== */

.metric-purple {
    background: linear-gradient(
        135deg,
        rgba(124, 58, 237, 0.20),
        rgba(139, 92, 246, 0.10)
    );

    border: 1px solid rgba(139, 92, 246, 0.70);

    box-shadow:
        0 8px 30px rgba(124, 58, 237, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}


/* ========================================
   BLUE CARD
======================================== */

.metric-blue {
    background: linear-gradient(
        135deg,
        rgba(37, 99, 235, 0.20),
        rgba(59, 130, 246, 0.09)
    );

    border: 1px solid rgba(59, 130, 246, 0.70);

    box-shadow:
        0 8px 30px rgba(37, 99, 235, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}


/* ========================================
   GREEN CARD
======================================== */

.metric-green {
    background: linear-gradient(
        135deg,
        rgba(16, 185, 129, 0.19),
        rgba(20, 184, 166, 0.08)
    );

    border: 1px solid rgba(45, 212, 191, 0.65);

    box-shadow:
        0 8px 30px rgba(16, 185, 129, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}


/* ========================================
   METRIC ICON
======================================== */

.metric-icon {
    width: 72px;
    height: 72px;

    flex-shrink: 0;

    display: flex;
    justify-content: center;
    align-items: center;

    border-radius: 18px;

    font-size: 34px;

    color: white;
}


/* Purple icon */

.metric-purple .metric-icon {
    background: linear-gradient(
        135deg,
        #8B5CF6,
        #6D28D9
    );

    box-shadow:
        0 8px 25px rgba(124, 58, 237, 0.25);
}


/* Blue icon */

.metric-blue .metric-icon {
    background: linear-gradient(
        135deg,
        #3B82F6,
        #2563EB
    );

    box-shadow:
        0 8px 25px rgba(37, 99, 235, 0.25);
}


/* Green icon */

.metric-green .metric-icon {
    background: linear-gradient(
        135deg,
        #2DD4BF,
        #0D9488
    );

    box-shadow:
        0 8px 25px rgba(13, 148, 136, 0.25);
}


    /* ========================================
    METRIC TEXT
    ======================================== */

    .metric-content {
        min-width: 0;
        position: relative;
        z-index: 2;
    }


    .metric-label {
        margin-bottom: 6px;

        font-size: 18px;
        font-weight: 600;

        color: rgba(248, 250, 252, 0.92);
    }


    .metric-value {
        font-size: 42px;
        font-weight: 500;

        line-height: 1.05;

        color: #FFFFFF;

        letter-spacing: -1px;
    }


    /* Dataset id puede ser más largo */
    .metric-dataset {
        font-size: 32px;

        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }


    /* ========================================
    CARD HOVER
    ======================================== */

    .metric-purple:hover {
        transform: translateY(-4px);

        border-color: rgba(167, 139, 250, 0.95);

        box-shadow:
            0 12px 32px rgba(124, 58, 237, 0.18),
            0 0 22px rgba(124, 58, 237, 0.12);
    }


    .metric-blue:hover {
        transform: translateY(-4px);

        border-color: rgba(96, 165, 250, 0.95);

        box-shadow:
            0 12px 32px rgba(37, 99, 235, 0.18),
            0 0 22px rgba(59, 130, 246, 0.12);
    }


    .metric-green:hover {
        transform: translateY(-4px);

        border-color: rgba(94, 234, 212, 0.95);

        box-shadow:
            0 12px 32px rgba(13, 148, 136, 0.18),
            0 0 22px rgba(45, 212, 191, 0.12);
    }


    /* ========================================
    SOURCE FINGERPRINT
    ======================================== */

    .source-fingerprint {
        margin-top: 20px;
        margin-bottom: 14px;

        font-size: 14px;

        color: rgba(203, 213, 225, 0.70);
    }


    .source-fingerprint span {
        margin-left: 6px;

        color: rgba(52, 211, 153, 0.85);

        font-family: monospace;

        overflow-wrap: anywhere;
    }
        

        </style>
        """,
        unsafe_allow_html=True,
    )