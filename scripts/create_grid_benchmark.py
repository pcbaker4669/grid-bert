import json
from pathlib import Path

OUTPUT = Path(
    r"D:\GridBERT\experiments\GridBERT-v0.1\grid_domain_benchmark.jsonl"
)

prompts = [

    # Markets
    ("markets", "PJM operates a wholesale electricity [MASK].", "market"),
    ("markets", "Electricity generators submit offers into the wholesale [MASK].", "market"),
    ("markets", "Locational marginal pricing reflects electricity [MASK].", "prices"),
    ("markets", "The capacity market helps ensure adequate generation [MASK].", "resources"),
    ("markets", "Generators compete to sell electricity in the regional [MASK].", "market"),

    # Reliability
    ("reliability", "NERC establishes reliability [MASK].", "standards"),
    ("reliability", "A shortage of generation can threaten grid [MASK].", "reliability"),
    ("reliability", "Grid operators must maintain system [MASK].", "reliability"),
    ("reliability", "Resource shortages may create reliability [MASK].", "risks"),
    ("reliability", "NERC evaluates the long-term reliability of the bulk power [MASK].", "system"),

    # Resource adequacy
    ("resource_adequacy", "The utility must maintain adequate generation [MASK].", "capacity"),
    ("resource_adequacy", "Resource adequacy requires sufficient generating [MASK].", "capacity"),
    ("resource_adequacy", "Peak demand requires adequate generation [MASK].", "resources"),
    ("resource_adequacy", "Planning reserves provide additional generation [MASK].", "capacity"),
    ("resource_adequacy", "Insufficient capacity can create a resource adequacy [MASK].", "risk"),

    # Transmission
    ("transmission", "Electricity moves across high-voltage transmission [MASK].", "lines"),
    ("transmission", "Transmission constraints can restrict power [MASK].", "flows"),
    ("transmission", "New transmission can increase transfer [MASK].", "capability"),
    ("transmission", "The transmission system connects generators with electrical [MASK].", "load"),
    ("transmission", "Transmission planning identifies future system [MASK].", "needs"),

    # Congestion
    ("congestion", "Transmission constraints can create grid [MASK].", "congestion"),
    ("congestion", "Congestion occurs when transmission capacity is [MASK].", "limited"),
    ("congestion", "Transmission congestion can increase electricity [MASK].", "prices"),
    ("congestion", "Congestion reflects constraints on electricity [MASK].", "flows"),
    ("congestion", "Grid upgrades can reduce transmission [MASK].", "congestion"),

    # Dispatch
    ("dispatch", "The grid operator dispatched additional generation [MASK].", "resources"),
    ("dispatch", "Generators are dispatched to meet electricity [MASK].", "demand"),
    ("dispatch", "Economic dispatch determines which generators should [MASK].", "operate"),
    ("dispatch", "The system operator dispatches generating [MASK].", "units"),
    ("dispatch", "Additional resources were dispatched to maintain system [MASK].", "reliability"),

    # Load
    ("load", "Electricity demand is commonly called electrical [MASK].", "load"),
    ("load", "Peak load represents the highest electricity [MASK].", "demand"),
    ("load", "System load increases when electricity demand [MASK].", "rises"),
    ("load", "Utilities forecast future electricity [MASK].", "demand"),
    ("load", "Data centers can significantly increase electricity [MASK].", "demand"),

    # Generation
    ("generation", "Power plants generate electrical [MASK].", "energy"),
    ("generation", "A generator converts energy into electricity for the [MASK].", "grid"),
    ("generation", "Generation resources supply electricity to meet [MASK].", "demand"),
    ("generation", "Retiring power plants reduces available generation [MASK].", "capacity"),
    ("generation", "New generating units can increase electricity [MASK].", "supply"),

    # Reserves / emergencies
    ("reserves", "Operating reserves provide backup generation during system [MASK].", "emergencies"),
    ("reserves", "Reserve capacity can respond when generation unexpectedly [MASK].", "fails"),
    ("reserves", "Grid operators maintain reserves to respond to unexpected [MASK].", "events"),
    ("reserves", "Emergency generation may be needed during capacity [MASK].", "shortages"),
    ("reserves", "A reliability emergency may require additional generation [MASK].", "resources"),

    # Institutions / policy
    ("institutions", "FERC regulates interstate transmission of electric [MASK].", "energy"),
    ("institutions", "NERC develops reliability standards for the bulk power [MASK].", "system"),
    ("institutions", "PJM coordinates electricity markets and grid [MASK].", "operations"),
    ("institutions", "DOE may issue emergency orders during electricity [MASK].", "emergencies"),
    ("institutions", "Regional transmission organizations coordinate wholesale electricity [MASK].", "markets"),
]

with open(OUTPUT, "w", encoding="utf-8") as f:
    for category, prompt, expected in prompts:
        f.write(json.dumps({
            "category": category,
            "prompt": prompt,
            "expected": expected
        }) + "\n")

print(f"Benchmark prompts: {len(prompts)}")
print(f"Saved: {OUTPUT}")