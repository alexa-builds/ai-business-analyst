
# Monthly sales data for a small business
import csv
import matplotlib.pyplot as plt

def calculate_metrics(sales, months):
    """Calculate key metrics from a list of monthly sales."""

    total_sales = sum(sales)
    average_sales = total_sales / len(sales)
    best_month = max(sales)
    worst_month = min(sales)

    first_month_sales = sales[0]
    last_month_sales = sales[-1]
    first_month_name = months[0]
    last_month_name = months[-1]

    change_percent = (
        ((last_month_sales - first_month_sales) / first_month_sales) * 100
        if first_month_sales != 0
        else None
    )
    sales_change = last_month_sales - first_month_sales
    return {
        "total_sales": total_sales,
        "average_sales": average_sales,
        "best_month": best_month,
        "worst_month": worst_month,
        "first_month_name": first_month_name,
        "last_month_name": last_month_name,
        "first_month_sales": first_month_sales,
        "last_month_sales": last_month_sales,
        "change_percent": change_percent,
        "sales_change": sales_change,
    }


sales = []
sales_from_csv = []
months = []

csv_filename = input("Enter CSV filename: ")

try:
    with open(csv_filename, "r", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "month" not in reader.fieldnames or "sales" not in reader.fieldnames:
            print("Error: CSV must contain 'month' and 'sales' columns.")
            raise SystemExit(1)

        for row in reader:
            sales_value = row["sales"]

            try:
                sales_value = float(sales_value)
                sales_from_csv.append(sales_value)
                months.append(row["month"])
            except (TypeError, ValueError):
                print("Invalid sales value for", row["month"], "- skipping row.")

except FileNotFoundError:
    print("Error: File not found. Please check the filename.")
    exit()

sales = sales_from_csv

if len(sales) == 0:
    print("Error: No valid sales data found.")
    exit()
best_index = sales.index(max(sales))
worst_index = sales.index(min(sales))
sales_difference = max(sales) - min(sales)
sales_total = sum(sales)
best_month_share = (max(sales) / sales_total) * 100 if sales_total > 0 else None
metrics = calculate_metrics(sales, months)

total_sales = sum(sales)
average_sales = total_sales / len(sales)
best_month = max(sales)
worst_month = min(sales)
first_month_sales = sales[0]
last_month_sales = sales[-1]
first_month_name = months[0]
last_month_name = months[-1]

change_percent = metrics["change_percent"]
sales_change = metrics["sales_change"]

print("BUSINESS SALES REPORT")
print("Months analysed:", len(months))
print("---------------------")
print("Total sales:", f'{metrics["total_sales"]:.2f}', "EUR")
print("Average monthly sales:", f'{metrics["average_sales"]:.2f}', "EUR")
print(f"Best month: {months[best_index]}: {max(sales):.2f} EUR")
print(f"Worst month: {months[worst_index]}: {min(sales):.2f} EUR")
print("Difference between best and worst months:", f"{sales_difference:.2f}", "EUR")
print("Best month share:", f"{best_month_share:.2f} %" if best_month_share is not None else "N/A (total sales are not positive)")
change_text = f"{change_percent:.2f} %" if change_percent is not None else "N/A (first sales value is zero)"
print("Sales change from", months[0], "to", months[-1] + ":", change_text)

if change_percent is None:
    print("Business trend: Change unavailable because starting sales are zero.")
elif change_percent > 0:
    print("Business trend: Sales increased.")
elif change_percent < 0:
    print("Business trend: Sales decreased.")
else:
    print("Business trend: Sales stayed the same.")

print("\nPERFORMANCE ASSESSMENT")


def assess_performance(change_percent):
    """Classify sales performance based on percentage change."""

    if change_percent is None:
        return "Change unavailable because starting sales are zero."
    elif change_percent > 10:
        return "Strong growth."
    elif change_percent > 0:
        return "Modest growth."
    elif change_percent == 0:
        return "Stable sales."
    else:
        return "Sales declined."


assessment = assess_performance(change_percent)
print("Assessment:", assessment)

plt.figure(figsize=(10, 5))

plt.plot(months, sales, marker="o")
trend = (
    [sales[0] + (sales[-1] - sales[0]) * i / (len(sales) - 1) for i in range(len(sales))]
    if len(sales) > 1
    else [sales[0]]
)

plt.plot(months, trend, linestyle="--", label="Overall trend")
plt.legend()

for month, sale in zip(months, sales):
    plt.annotate(
        f"€{sale}",
        (month, sale),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
    )

plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales (EUR)")

plt.grid(True)

plt.savefig("sales_chart.png", dpi=300, bbox_inches="tight")
plt.show()
