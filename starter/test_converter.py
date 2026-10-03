from sql_converter import convert_sql


header = [
    "Player",
    "No.",
    "Nationality",
    "Position",
    "Years in Toronto",
    "School/Club Team"
]

sql = "select <c2> where <c0> = terrence ross"

result = convert_sql(
    sql,
    header
)

print(result)