"""What the old marker did on the layouts the checkpoint's own parser accepts."""

OLD_PREFIX = "\n\n<｜DSML｜ calls"

BLOCK = (
    '<｜DSML｜ calls>\n'
    '<｜DSML｜ invoke name="lookup">\n'
    '<｜DSML｜ parameter name="query" string="true">value</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n'
    '</｜DSML｜ calls>'
)

cases = {
    "double_newline (trained layout)": "summary\n\n" + BLOCK,
    "single_newline": "summary\n" + BLOCK,
    "no_newline": "summary" + BLOCK,
    "block_first": BLOCK,
    "spaced_tags": "summary\n\n<｜DSML｜ calls> <｜DSML｜ invoke name=\"lookup\"> <｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter> </｜DSML｜ invoke> </｜DSML｜ calls>",
}

for name, text in cases.items():
    start = text.find(OLD_PREFIX)
    detected = start >= 0
    leaked = not detected and "<｜DSML｜" in text
    print(f"  {name:<34} old_detect={int(detected)}  raw_markup_in_content={int(leaked)}")
