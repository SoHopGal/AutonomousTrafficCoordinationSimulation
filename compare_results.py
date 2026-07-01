import json

with open("fixed_results.json") as f:
    fixed = json.load(f)

with open("adaptive_results.json") as f:
    adaptive = json.load(f)


print(f"{'Scenario':10} {'Metric':15} {'Fixed':10} {'Adaptive':10}")

print("-" * 50)

for f, a in zip(fixed, adaptive):
    print(
        f"{f['scenario']:10} "
        f"{'Avg Travel':15} "
        f"{f['avg_travel']:<10.2f} "
        f"{a['avg_travel']:<10.2f}"
    )

    print(
        f"{'':10} "
        f"{'Avg Wait':15} "
        f"{f['avg_wait']:<10.2f} "
        f"{a['avg_wait']:<10.2f}"
    )

    print(
        f"{'':10} "
        f"{'Collisions':15} "
        f"{f['collisions']:<10} "
        f"{a['collisions']:<10}"
    )

    print()
