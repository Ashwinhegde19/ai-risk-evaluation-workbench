"""Generate placeholder dashboard screenshots for README."""

from PIL import Image, ImageDraw, ImageFont

PAGES = [
    ("overview", "Overview", "Radar chart of mean safety scores across 7 risk dimensions\nKPI tiles: overall score, models evaluated, red-team success rate"),
    ("comparison", "Model Comparison", "Side-by-side radar charts for two models\nPer-dimension score table with response diffs"),
    ("redteam", "Red-Team Results", "Attack-tree graph (Graphviz) showing multi-turn escalation\nTurn-by-turn drill-down with strategy chain and scores"),
    ("compliance", "Compliance", "EU AI Act / NIST AI RMF / ISO 42001 findings\nGap analysis with control IDs and severity\nPDF/JSON export buttons"),
    ("trends", "Trends", "Historical score tracking across evaluation runs\nLine charts per dimension with regression markers"),
]

WIDTH, HEIGHT = 1200, 700
BG_COLOR = (250, 250, 250)
HEADER_COLOR = (30, 30, 30)
TEXT_COLOR = (80, 80, 80)
ACCENT_COLOR = (0, 122, 204)

def create_placeholder(filename, title, description):
    """Create a clean labeled placeholder image."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Try to load a nice font, fall back to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        desc_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        desc_font = ImageFont.load_default()

    # Draw header bar
    draw.rectangle([0, 0, WIDTH, 100], fill=ACCENT_COLOR)

    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (WIDTH - title_width) // 2
    draw.text((title_x, 25), title, fill=(255, 255, 255), font=title_font)

    # Draw description lines
    y_offset = 200
    for line in description.split("\n"):
        desc_bbox = draw.textbbox((0, 0), line, font=desc_font)
        desc_width = desc_bbox[2] - desc_bbox[0]
        desc_x = (WIDTH - desc_width) // 2
        draw.text((desc_x, y_offset), line, fill=TEXT_COLOR, font=desc_font)
        y_offset += 50

    # Draw placeholder box
    box_margin = 100
    draw.rectangle(
        [box_margin, 350, WIDTH - box_margin, HEIGHT - 50],
        outline=(200, 200, 200),
        width=3
    )
    draw.text(
        (WIDTH // 2 - 100, 480),
        "[ Dashboard Preview ]",
        fill=(150, 150, 150),
        font=desc_font
    )

    img.save(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    for filename, title, desc in PAGES:
        create_placeholder(f"/home/ashwin/Projects/ai-risk-evaluation-workbench/docs/screenshots/{filename}.png", title, desc)
