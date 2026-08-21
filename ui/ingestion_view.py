"""
UI del Ingestion Agent.
Dibuja: tarjetas de salud, issues, SQL/indices/alertas y el chat.
Consume el reporte declarativo de crew.ingestion_agent.ingestion_agent.
"""
import pandas as pd
import streamlit as st

from crew.ingestion_agent.ingestion_agent import answer_question, build_ingestion_report

_STATUS_ICON = {"good": "🟢", "warn": "🟡", "bad": "🔴"}


def render_ingestion(df, table_name: str = "ingested_data") -> dict:
    report = build_ingestion_report(df, table_name)

    # =========================================================
    # 1. TARJETAS DE SALUD
    # =========================================================

    st.subheader("Data health")

    cards = report["cards"]
    cols = st.columns(len(cards))

    for col, card in zip(cols, cards):
        col.metric(
            f"{_STATUS_ICON[card['status']]} {card['label']}",
            card["value"],
        )
        col.caption(card["detail"])


    # =========================================================
    # 2. COLUMN PROFILE
    # =========================================================

    st.markdown("---")
    st.subheader("Column profile")

    profile_df = pd.DataFrame(
        [
            {
                "Column": column["name"],
                "Data type": column["dtype"],
                "Unique values": column["unique"],
                "Missing values": column["nulls"],
            }
            for column in report["columns"]
        ]
    )

    st.dataframe(
        profile_df,
        use_container_width=True,
        hide_index=True,
    )


    # =========================================================
    # 3. ISSUES DETECTADOS
    # =========================================================

    issues = report["issues"]

    st.subheader(
        f"Detected issues ({len(issues)})"
    )

    if not issues:
        st.success(
            "No data-quality issues detected. 🎉"
        )

    for iss in issues:
        with st.container(border=True):
            st.markdown(
                f"**{iss['title']}** "
                f"&nbsp;·&nbsp; "
                f"severity: `{iss['severity']}` "
                f"&nbsp;·&nbsp; "
                f"affected rows: {iss['count']}"
            )

            st.caption(
                f"{iss['detail']} "
                f"→ *suggested: "
                f"{iss['suggested_action']}*"
            )


    # =========================================================
    # 4. SQL / INDICES / ALERTAS
    # =========================================================

    table = report["meta"]["table_name"]

    with st.expander(
        "🗄️ SQL script — CREATE TABLE (SQL Server)"
    ):
        st.code(
            report["sql"]["create_table"],
            language="sql",
        )

        if report["sql"]["index_ddl"]:
            st.markdown("**Index DDL:**")

            st.code(
                "\n".join(
                    report["sql"]["index_ddl"]
                ),
                language="sql",
            )

    with st.expander("🔑 Suggested indexes"):
        st.write(
            "Primary-key candidate: "
            f"**{report['primary_key'] or 'none found'}**"
        )

        for i in report["indexes"]:
            nombre = i["name"].replace(
                "{table}",
                table,
            )

            st.markdown(
                f"- `{nombre}` — "
                f"**{i['type']}** on "
                f"{', '.join(i['columns'])}  \n"
                f"  <small>{i['rationale']}</small>",
                unsafe_allow_html=True,
            )

    with st.expander(
        "🔔 Suggested data-quality alerts"
    ):
        for a in report["alerts"]:
            st.markdown(
                f"- **{a['name']}** — "
                f"{a['type']} / `{a['severity']}`  \n"
                f"  condition: `{a['condition']}`  \n"
                f"  <small>{a['rationale']}</small>",
                unsafe_allow_html=True,
            )


    # =========================================================
    # 5. CHAT
    # =========================================================

    st.subheader("💬 Ask the ingestion agent")

    if "ingestion_chat" not in st.session_state:
        st.session_state["ingestion_chat"] = []

    for role, msg in st.session_state["ingestion_chat"]:
        with st.chat_message(role):
            st.markdown(msg)

    pregunta = st.chat_input(
        "Ask about your data: rows, columns, "
        "missing values, duplicates, types..."
    )

    if pregunta:
        st.session_state[
            "ingestion_chat"
        ].append(("user", pregunta))

        with st.chat_message("user"):
            st.markdown(pregunta)

        respuesta = answer_question(
            df,
            pregunta,
        )

        st.session_state[
            "ingestion_chat"
        ].append(
            ("assistant", respuesta)
        )

        with st.chat_message("assistant"):
            st.markdown(respuesta)

    return report