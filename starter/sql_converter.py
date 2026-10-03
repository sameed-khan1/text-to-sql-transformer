import re


def convert_sql(sql_text, header):
    sql_text = sql_text.lower().strip()

    select_match = re.search(
        r"select\s+(?:(max|min|count|sum|avg)\s+)?<c(\d+)>",
        sql_text
    )

    if not select_match:
        return None

    agg = select_match.group(1)
    select_col = int(select_match.group(2))

    if select_col >= len(header):
        return None

    select_name = header[select_col]

    if agg:
        select_part = f"{agg.upper()}({select_name})"
    else:
        select_part = select_name

    sql = f"SELECT {select_part}"

    where_match = re.search(
        r"where\s+<c(\d+)>\s*(=|>|<)\s*(.*?)(?:\s+and\s+|$)",
        sql_text
    )

    if where_match:
        col = int(where_match.group(1))
        op = where_match.group(2)
        value = where_match.group(3).strip()

        if col >= len(header):
            return None

        column_name = header[col]

        value = value.replace("'", "''")

        sql += (
            f" WHERE {column_name} "
            f"{op} '{value}'"
        )

    return sql