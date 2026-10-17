#!/usr/bin/env python3
"""Pixel-grid font tooling for the Deadass Simple typeface.

Glyph sources are plain-text pixel grids, one `.glyph` file per Unicode codepoint
(`Glyphs/U+0041.glyph` is "A"). `Metrics.toml` holds the names and the vertical layout.

Subcommands:
  extract-logo  read the logo PNG and write a .glyph file for each letter it contains
  import-svg    read SVG glyphs exported from Sketch and write a .glyph file for each new letter
  build         turn the .glyph files into a TTF and an OTF
  verify        render built fonts with FreeType and compare every pixel to the .glyph grids
  specimen      render the specimen text (SPECIMEN_LINES) from a built font as a PNG, in the logo's own colors

.glyph format: optional `;` comment lines, header lines `advance: <pixels>` and
`top: <pixels from the baseline up to the top edge of the first row>`, then pixel rows
made only of `#` (filled) and `.` (empty). A glyph with no rows (space) needs no `top`.
"""

import argparse
import re
import sys
import tomllib
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fontTools import agl
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

GLYPH_FILE_PATTERN = re.compile(r"^U\+([0-9A-Fa-f]{4,6})\.glyph$")
NOTDEF_FILE_NAME = "notdef.glyph"

# The logo is a square of LOGO_CELL_COUNT x LOGO_CELL_COUNT cells. Letters sit in a 4 x 4 grid of
# slots, each LOGO_GLYPH_SIZE cells square, LOGO_PITCH cells apart (a one cell gap between them).
LOGO_CELL_COUNT = 48
LOGO_GLYPH_SIZE = 7
LOGO_PITCH = 8
LOGO_FIRST_SLOT_COLUMN = 8
LOGO_FIRST_BAND_ROW = 9
# One string per band (row of letters). A space marks a slot that holds the dot, not a letter.
LOGO_BANDS = ("DEAD", " ASS", " SIM", " PLE")
LOGO_LOCKUP_LINES = ("DEAD", "•ASS", "•SIM", "•PLE")  # the same bands as typed text, with the dots as U+2022


@dataclass(frozen=True)
class Metrics:
    family_name: str
    style_name: str
    version: str
    units_per_pixel: int
    cap_rows: int
    rows_above_cap: int
    rows_below_baseline: int
    # optional name table texts; an empty string leaves that name record out
    copyright: str = ""
    license: str = ""
    manufacturer: str = ""
    designer: str = ""
    vendor_url: str = ""
    designer_url: str = ""

    @property
    def ascent(self) -> int:
        return (self.cap_rows + self.rows_above_cap) * self.units_per_pixel

    @property
    def descent(self) -> int:
        return self.rows_below_baseline * self.units_per_pixel

    @property
    def units_per_em(self) -> int:
        return self.ascent + self.descent


@dataclass(frozen=True)
class Glyph:
    codepoint: int | None  # None for .notdef
    advance: int  # pixels
    top: int  # pixels from the baseline up to the top edge of the first row
    rows: tuple[str, ...]

    def filled_cells(self) -> set[tuple[int, int]]:
        """Unit squares as (x, y) lower-left corners, with y counted upward from the baseline."""
        return {
            (column, self.top - row - 1)
            for row, line in enumerate(self.rows)
            for column, character in enumerate(line)
            if "#" == character
        }


def load_metrics(path: Path) -> Metrics:
    with path.open("rb") as file:
        data = tomllib.load(file)
    names = data["names"]
    grid = data["grid"]
    return Metrics(
        family_name=names["family"],
        style_name=names["style"],
        version=names["version"],
        units_per_pixel=grid["units_per_pixel"],
        cap_rows=grid["cap_rows"],
        rows_above_cap=grid["rows_above_cap"],
        rows_below_baseline=grid["rows_below_baseline"],
        copyright=names.get("copyright", ""),
        license=names.get("license", ""),
        manufacturer=names.get("manufacturer", ""),
        designer=names.get("designer", ""),
        vendor_url=names.get("vendor_url", ""),
        designer_url=names.get("designer_url", ""),
    )


def parse_glyph(path: Path) -> Glyph:
    codepoint = None
    if NOTDEF_FILE_NAME != path.name:
        match = GLYPH_FILE_PATTERN.match(path.name)
        if None is match:
            raise ValueError(f"{path.name}: file name must be U+XXXX.glyph or {NOTDEF_FILE_NAME}")
        codepoint = int(match.group(1), 16)

    header: dict[str, str] = {}
    rows: list[str] = []
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if "" == line or line.startswith(";"):
            continue
        if re.fullmatch(r"[#.]+", line):
            rows.append(line)
        elif 0 != len(rows):
            raise ValueError(f"{path.name}:{number}: header line after the pixel rows")
        else:
            key, separator, value = line.partition(":")
            if "" == separator:
                raise ValueError(f"{path.name}:{number}: expected `key: value`, got `{line}`")
            header[key.strip()] = value.strip()

    unknown = set(header) - {"advance", "top"}
    if 0 != len(unknown):
        raise ValueError(f"{path.name}: unknown header keys {sorted(unknown)}")
    if "advance" not in header:
        raise ValueError(f"{path.name}: missing `advance:`")
    if 0 != len(rows) and "top" not in header:
        raise ValueError(f"{path.name}: has pixel rows but no `top:`")
    if 1 != len({len(row) for row in rows}) and 0 != len(rows):
        raise ValueError(f"{path.name}: pixel rows differ in width")

    advance = int(header["advance"])
    if 0 != len(rows) and advance < len(rows[0]):
        raise ValueError(f"{path.name}: rows are {len(rows[0])} wide but advance is {advance}")
    return Glyph(codepoint, advance, int(header.get("top", "0")), tuple(rows))


def load_glyphs(directory: Path) -> dict[int | None, Glyph]:
    glyphs: dict[int | None, Glyph] = {}
    for path in sorted(directory.glob("*.glyph")):
        glyph = parse_glyph(path)
        if glyph.codepoint in glyphs:
            raise ValueError(f"{path.name}: codepoint is already defined by another file")
        glyphs[glyph.codepoint] = glyph
    return glyphs


def glyph_name(codepoint: int | None) -> str:
    if None is codepoint:
        return ".notdef"
    name = agl.UV2AGL.get(codepoint)
    if None is not name:
        return name
    return f"uni{codepoint:04X}" if codepoint <= 0xFFFF else f"u{codepoint:X}"


def simplify(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop points that sit in the middle of a straight run."""
    kept = []
    for index, here in enumerate(points):
        before = points[index - 1]
        after = points[(index + 1) % len(points)]
        cross = (here[0] - before[0]) * (after[1] - here[1]) - (here[1] - before[1]) * (after[0] - here[0])
        if 0 != cross:
            kept.append(here)
    return kept


def trace_contours(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Closed polygons around the unit squares in `cells`.

    Every square contributes four clockwise edges; an edge shared by two squares appears once in
    each direction and cancels. What remains are the outlines: clockwise around filled areas and
    counterclockwise around holes. Where two squares touch only at a corner, the trace turns right
    so each square keeps its own outline.
    """
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for x, y in cells:
        for edge in (
            ((x, y), (x, y + 1)),
            ((x, y + 1), (x + 1, y + 1)),
            ((x + 1, y + 1), (x + 1, y)),
            ((x + 1, y), (x, y)),
        ):
            reverse = (edge[1], edge[0])
            if reverse in edges:
                edges.remove(reverse)
            else:
                edges.add(edge)

    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for start, end in sorted(edges):
        outgoing.setdefault(start, []).append(end)

    unvisited = set(edges)
    contours = []
    while 0 != len(unvisited):
        first = min(unvisited)
        unvisited.remove(first)
        points = [first[0], first[1]]
        while True:
            previous_point, point = points[-2], points[-1]
            direction = (point[0] - previous_point[0], point[1] - previous_point[1])
            chosen = min(
                outgoing[point],
                key=lambda end: direction[0] * (end[1] - point[1]) - direction[1] * (end[0] - point[0]),
            )
            edge = (point, chosen)
            if first == edge:
                break
            if edge not in unvisited:
                raise RuntimeError("contour tracing reached an edge it had already used")
            unvisited.remove(edge)
            points.append(chosen)
        points.pop()  # the last point repeats the first
        contours.append(simplify(points))
    return contours


def glyph_contours(glyph: Glyph, metrics: Metrics) -> list[list[tuple[int, int]]]:
    scale = metrics.units_per_pixel
    return [
        [(x * scale, y * scale) for x, y in contour]
        for contour in trace_contours(glyph.filled_cells())
    ]


def draw_contours(pen, contours) -> None:
    for contour in contours:
        pen.moveTo(contour[0])
        for point in contour[1:]:
            pen.lineTo(point)
        pen.closePath()


def font_names(metrics: Metrics) -> dict:
    postscript_name = f"{metrics.family_name}-{metrics.style_name}".replace(" ", "")
    names = dict(
        familyName=dict(en=metrics.family_name),
        styleName=dict(en=metrics.style_name),
        uniqueFontIdentifier=f"{postscript_name};{metrics.version}",
        fullName=f"{metrics.family_name} {metrics.style_name}",
        psName=postscript_name,
        version=f"Version {metrics.version}",
    )
    optional = dict(
        copyright=metrics.copyright,
        manufacturer=metrics.manufacturer,
        designer=metrics.designer,
        vendorURL=metrics.vendor_url,
        designerURL=metrics.designer_url,
        licenseDescription=metrics.license,
    )
    names.update({key: value for key, value in optional.items() if "" != value})
    return names


def build_font(metrics: Metrics, glyph_order: list[str], glyphs_by_name: dict[str, Glyph],
               cmap: dict[int, str], is_ttf: bool) -> FontBuilder:
    builder = FontBuilder(metrics.units_per_em, isTTF=is_ttf)
    builder.setupGlyphOrder(glyph_order)
    builder.setupCharacterMap(cmap)
    names = font_names(metrics)

    horizontal_metrics = {}
    outlines = {}
    for name in glyph_order:
        glyph = glyphs_by_name[name]
        contours = glyph_contours(glyph, metrics)
        advance = glyph.advance * metrics.units_per_pixel
        left_bearing = min((x for contour in contours for x, _ in contour), default=0)
        horizontal_metrics[name] = (advance, left_bearing)
        outlines[name] = (advance, contours)

    if is_ttf:
        glyf = {}
        for name, (_, contours) in outlines.items():
            pen = TTGlyphPen(None)
            draw_contours(pen, contours)
            glyf[name] = pen.glyph()
        builder.setupGlyf(glyf)
    else:
        char_strings = {}
        for name, (advance, contours) in outlines.items():
            pen = T2CharStringPen(advance, None)
            # CFF wants outer contours counterclockwise, the reverse of TrueType
            draw_contours(pen, [contour[::-1] for contour in contours])
            char_strings[name] = pen.getCharString()
        builder.setupCFF(names["psName"], {"FullName": names["fullName"]}, char_strings, {})

    monospaced = 1 == len({glyph.advance for glyph in glyphs_by_name.values()})
    builder.setupHorizontalMetrics(horizontal_metrics)
    builder.setupHorizontalHeader(ascent=metrics.ascent, descent=-metrics.descent, lineGap=0)
    builder.setupNameTable(names)
    builder.setupOS2(
        version=4,  # fsSelection bit 7 (USE_TYPO_METRICS) needs OS/2 version 4 or later
        sTypoAscender=metrics.ascent,
        sTypoDescender=-metrics.descent,
        sTypoLineGap=0,
        usWinAscent=metrics.ascent,
        usWinDescent=metrics.descent,
        sCapHeight=metrics.cap_rows * metrics.units_per_pixel,
        usWeightClass=400,
        usWidthClass=5,
        fsType=0,
        fsSelection=0x40 | 0x80,  # REGULAR | USE_TYPO_METRICS
    )
    builder.setupPost(
        keepGlyphNames=is_ttf,
        isFixedPitch=int(monospaced),
        underlinePosition=-metrics.units_per_pixel,
        underlineThickness=metrics.units_per_pixel,
    )
    builder.updateHead(fontRevision=float(metrics.version))
    return builder


def build_fonts(source_directory: Path, output_directory: Path) -> list[Path]:
    metrics = load_metrics(source_directory / "Metrics.toml")
    glyphs = load_glyphs(source_directory / "Glyphs")
    if None not in glyphs:
        raise SystemExit(f"Glyphs/{NOTDEF_FILE_NAME} is required")

    codepoints = sorted(codepoint for codepoint in glyphs if None is not codepoint)
    glyph_order = [".notdef"] + [glyph_name(codepoint) for codepoint in codepoints]
    if len(set(glyph_order)) != len(glyph_order):
        raise SystemExit("two codepoints map to the same glyph name")
    glyphs_by_name = {glyph_name(codepoint): glyphs[codepoint] for codepoint in glyphs}
    cmap = {codepoint: glyph_name(codepoint) for codepoint in codepoints}

    output_directory.mkdir(parents=True, exist_ok=True)
    written = []
    for extension, is_ttf in (("ttf", True), ("otf", False)):
        path = output_directory / f"{metrics.family_name} {metrics.style_name}.{extension}"
        build_font(metrics, glyph_order, glyphs_by_name, cmap, is_ttf).save(str(path))
        reread = TTFont(str(path))
        print(
            f"wrote {path} ({path.stat().st_size} bytes): {len(reread.getGlyphOrder())} glyphs, "
            f"{len(reread.getBestCmap())} mapped codepoints, unitsPerEm {reread['head'].unitsPerEm}, "
            f"ascent {reread['hhea'].ascent}, descent {reread['hhea'].descent}"
        )
        written.append(path)
    return written


def lockup_lines():
    """Yield (text, first column, baseline row) for each line of the logo lockup, in cells."""
    for band, text in enumerate(LOGO_LOCKUP_LINES):
        baseline_row = LOGO_FIRST_BAND_ROW + band * LOGO_PITCH + LOGO_GLYPH_SIZE
        yield text, LOGO_FIRST_SLOT_COLUMN, baseline_row


def expected_lockup(glyphs: dict[int | None, Glyph], pixel_size: int) -> np.ndarray:
    """The lockup drawn straight from the .glyph grids, `pixel_size` screen pixels per grid pixel."""
    side = LOGO_CELL_COUNT * pixel_size
    drawn = np.zeros((side, side), dtype=bool)
    for text, column, baseline_row in lockup_lines():
        origin = column
        for character in text:
            glyph = glyphs[ord(character)]
            for row, line in enumerate(glyph.rows):
                for offset, mark in enumerate(line):
                    if "#" == mark:
                        top = (baseline_row - glyph.top + row) * pixel_size
                        left = (origin + offset) * pixel_size
                        drawn[top:top + pixel_size, left:left + pixel_size] = True
            origin += glyph.advance
    return drawn


def render_lockup(font_path: Path, metrics: Metrics, pixel_size: int) -> np.ndarray:
    """The lockup rendered by FreeType (through Pillow) from a built font file, as 8-bit coverage."""
    em_size = pixel_size * metrics.units_per_em // metrics.units_per_pixel
    font = ImageFont.truetype(str(font_path), em_size)
    side = LOGO_CELL_COUNT * pixel_size
    image = Image.new("L", (side, side), 0)
    draw = ImageDraw.Draw(image)
    for text, column, baseline_row in lockup_lines():
        draw.text((column * pixel_size, baseline_row * pixel_size), text, font=font, fill=255, anchor="ls")
    return np.asarray(image)


def verify_fonts(source_directory: Path, font_paths: list[Path], pixel_size: int) -> int:
    metrics = load_metrics(source_directory / "Metrics.toml")
    glyphs = load_glyphs(source_directory / "Glyphs")
    expected = expected_lockup(glyphs, pixel_size)
    failures = 0
    for font_path in font_paths:
        rendered = render_lockup(font_path, metrics, pixel_size)
        mismatched = int(np.count_nonzero((rendered > 127) != expected))
        partial = int(np.count_nonzero((0 < rendered) & (255 > rendered)))
        passed = 0 == mismatched and 0 == partial
        failures += 0 if passed else 1
        print(
            f"{'OK      ' if passed else 'MISMATCH'} {font_path.name}: {expected.shape[0]}x{expected.shape[1]} px render at "
            f"{pixel_size} px per grid pixel, {mismatched} pixels differ from the grids, "
            f"{partial} edge pixels not fully covered"
        )
    return 0 if 0 == failures else 1


def read_logo(path: Path):
    """Sample the logo PNG as a grid of cells. Returns (cells, ambiguous count, background, foreground)."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width != height:
        raise SystemExit(f"{path.name}: expected a square image, got {width}x{height}")
    pixels = np.asarray(image)
    is_light = pixels.min(axis=2) > 127
    cell_size = width / LOGO_CELL_COUNT
    fractions = np.zeros((LOGO_CELL_COUNT, LOGO_CELL_COUNT))
    for row in range(LOGO_CELL_COUNT):
        for column in range(LOGO_CELL_COUNT):
            # stay two pixels inside each cell so edge rounding cannot bleed in
            top, bottom = round(row * cell_size) + 2, round((row + 1) * cell_size) - 2
            left, right = round(column * cell_size) + 2, round((column + 1) * cell_size) - 2
            fractions[row, column] = is_light[top:bottom, left:right].mean()
    ambiguous = int(np.count_nonzero((fractions > 0.02) & (fractions < 0.98)))
    cells = fractions > 0.5
    first_row, first_column = np.argwhere(cells)[0]
    center_y = int((first_row + 0.5) * cell_size)
    center_x = int((first_column + 0.5) * cell_size)
    background = tuple(int(value) for value in pixels[0, 0])
    foreground = tuple(int(value) for value in pixels[center_y, center_x])
    print(
        f"{path.name}: {width}x{height} px, {LOGO_CELL_COUNT}x{LOGO_CELL_COUNT} cells of {cell_size} px, "
        f"background #{bytes(background).hex().upper()}, foreground #{bytes(foreground).hex().upper()}, "
        f"{len(np.unique(pixels.reshape(-1, 3), axis=0))} distinct colors in the file, "
        f"{ambiguous} ambiguous cells"
    )
    return cells, ambiguous, background, foreground


def slot_slices(band: int, slot: int) -> tuple[slice, slice]:
    row = LOGO_FIRST_BAND_ROW + band * LOGO_PITCH
    column = LOGO_FIRST_SLOT_COLUMN + slot * LOGO_PITCH
    return slice(row, row + LOGO_GLYPH_SIZE), slice(column, column + LOGO_GLYPH_SIZE)


def format_glyph_file(comment: str, advance: int, top: int, grid: np.ndarray) -> str:
    lines = [f"; {comment}", f"advance: {advance}", f"top: {top}"]
    lines += ["".join("#" if filled else "." for filled in row) for row in grid]
    return "\n".join(lines) + "\n"


def extract_logo(source_directory: Path, logo_path: Path, overwrite: bool) -> int:
    cells, ambiguous, _, _ = read_logo(logo_path)
    problems = []
    if 0 != ambiguous:
        problems.append(f"{ambiguous} cells are neither clearly filled nor clearly empty")

    canonical: dict[str, np.ndarray] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for band, letters in enumerate(LOGO_BANDS):
        for slot, letter in enumerate(letters):
            if " " == letter:
                continue
            grid = cells[slot_slices(band, slot)]
            if letter not in canonical:
                canonical[letter] = grid
                first_seen[letter] = (band, slot)
            elif not np.array_equal(canonical[letter], grid):
                problems.append(f"{letter} at band {band} slot {slot} differs from its first copy {first_seen[letter]}")

    copies = sum(1 for letters in LOGO_BANDS for letter in letters if " " != letter)
    print(f"{copies} letters in the logo, {len(canonical)} distinct: {''.join(sorted(canonical))}")

    rebuilt = np.zeros_like(cells)
    for band, letters in enumerate(LOGO_BANDS):
        for slot, letter in enumerate(letters):
            if " " != letter:
                rebuilt[slot_slices(band, slot)] = canonical[letter]
    leftover = cells & ~rebuilt
    missing = rebuilt & ~cells
    dot_region = np.zeros_like(cells)
    for band, letters in enumerate(LOGO_BANDS):
        for slot, letter in enumerate(letters):
            if " " == letter:
                rows, columns = slot_slices(band, slot)
                dot_region[rows, columns] = True
                found = np.argwhere(leftover[rows, columns])
                if 0 == len(found):
                    print(f"band {band} slot {slot}: no dot")
                else:
                    (top, left), (bottom, right) = found.min(axis=0), found.max(axis=0)
                    print(
                        f"band {band} slot {slot}: dot of {len(found)} cells, rows {top}-{bottom} and "
                        f"columns {left}-{right} within the {LOGO_GLYPH_SIZE}x{LOGO_GLYPH_SIZE} slot"
                    )
    unexplained = int(np.count_nonzero(leftover & ~dot_region)) + int(np.count_nonzero(missing))
    print(f"cells not explained by the letters or the dots: {unexplained}")
    if 0 != unexplained:
        problems.append(f"{unexplained} cells are not explained by the letters or the dots")

    glyph_directory = source_directory / "Glyphs"
    glyph_directory.mkdir(parents=True, exist_ok=True)
    for letter in sorted(canonical):
        path = glyph_directory / f"U+{ord(letter):04X}.glyph"
        if path.exists() and not overwrite:
            print(f"skipped {path.name}: exists (pass --overwrite to replace it)")
            continue
        band, slot = first_seen[letter]
        comment = f"{letter}, cut from the logo (band {band}, slot {slot})"
        path.write_text(format_glyph_file(comment, LOGO_PITCH, LOGO_GLYPH_SIZE, canonical[letter]), encoding="utf-8")
        print(f"wrote {path.name}")
        for row in canonical[letter]:
            print("    " + "".join("#" if filled else "." for filled in row))

    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 0 if 0 == len(problems) else 1


SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
SVG_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
SVG_SHAPES_WE_CANNOT_READ = {"rect", "circle", "ellipse", "line", "polyline"}


def whole_number(text: str) -> int:
    value = float(text)
    if value != round(value):
        raise ValueError(f"coordinate {text} is not a whole number of grid pixels")
    return round(value)


def svg_outlines(element) -> list[list[tuple[int, int]]]:
    """The closed outlines of one <polygon> or <path> (absolute M, L, H, V, Z only) as integer points."""
    tag = element.tag.removeprefix(SVG_NAMESPACE)
    if "polygon" == tag:
        numbers = [whole_number(text) for text in SVG_NUMBER.findall(element.get("points", ""))]
        if 0 != len(numbers) % 2:
            raise ValueError("polygon has an odd number of coordinates")
        return [list(zip(numbers[0::2], numbers[1::2]))]

    outlines: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for command, arguments in re.findall(r"([A-Za-z])([^A-Za-z]*)", element.get("d", "")):
        numbers = [whole_number(text) for text in SVG_NUMBER.findall(arguments)]
        if command in "ML":
            if 0 == len(numbers) or 0 != len(numbers) % 2:
                raise ValueError(f"path command {command} needs pairs of coordinates")
            if "M" == command:
                if 0 != len(current):
                    outlines.append(current)
                current = []
            current += list(zip(numbers[0::2], numbers[1::2]))
        elif command in "HV":
            if 0 == len(current):
                raise ValueError(f"path command {command} before any point")
            for number in numbers:
                x, y = current[-1]
                current.append((number, y) if "H" == command else (x, number))
        elif command in "Zz":
            if 0 != len(current):
                outlines.append(current)
            current = []
        else:
            raise ValueError(f"path command {command} is not supported (only absolute M, L, H, V, Z)")
    if 0 != len(current):
        outlines.append(current)
    return outlines


def fill_outlines(outlines, width: int, height: int, fill_rule: str) -> np.ndarray:
    """Which unit cells of a width x height canvas the outlines fill, sampled at each cell's center."""
    for points in outlines:
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
            if x1 != x2 and y1 != y2:
                raise ValueError(f"diagonal edge ({x1}, {y1}) to ({x2}, {y2}); pixel glyphs need horizontal and vertical edges")
    filled = np.zeros((height, width), dtype=bool)
    for row in range(height):
        for column in range(width):
            sample_x, sample_y = column + 0.5, row + 0.5
            crossings = 0
            winding = 0
            for points in outlines:
                for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
                    # a horizontal ray to the right of the sample point, crossing this edge
                    if (y1 <= sample_y) != (y2 <= sample_y) and x1 > sample_x:
                        crossings += 1
                        winding += 1 if y2 > y1 else -1
            filled[row, column] = (1 == crossings % 2) if "evenodd" == fill_rule else (0 != winding)
    return filled


def read_svg_canvas(path: Path) -> np.ndarray:
    """Every filled cell of one exported SVG, as a boolean grid the size of its viewBox."""
    root = ElementTree.parse(path).getroot()
    view_box = [whole_number(text) for text in SVG_NUMBER.findall(root.get("viewBox", ""))]
    if 4 != len(view_box) or (0, 0) != tuple(view_box[:2]):
        raise ValueError(f"expected a viewBox starting at 0 0, got `{root.get('viewBox')}`")
    width, height = view_box[2], view_box[3]
    canvas = np.zeros((height, width), dtype=bool)

    def walk(element, fill: str, fill_rule: str) -> None:
        nonlocal canvas
        fill = element.get("fill", fill)
        fill_rule = element.get("fill-rule", fill_rule)
        tag = element.tag.removeprefix(SVG_NAMESPACE)
        if tag in SVG_SHAPES_WE_CANNOT_READ:
            raise ValueError(f"<{tag}> is not supported; use polygons or paths with straight edges")
        if tag in ("polygon", "path") and "none" != fill:
            canvas |= fill_outlines(svg_outlines(element), width, height, fill_rule)
        for child in element:
            walk(child, fill, fill_rule)

    walk(root, "#000000", "nonzero")
    return canvas


def import_svg(source_directory: Path, svg_directory: Path, left_margin: int, glyph_width: int, overwrite: bool) -> int:
    metrics = load_metrics(source_directory / "Metrics.toml")
    top_margin, glyph_height = metrics.rows_above_cap, metrics.cap_rows
    canvas_height = metrics.rows_above_cap + metrics.cap_rows + metrics.rows_below_baseline
    glyph_directory = source_directory / "Glyphs"

    problems: list[str] = []
    imported: dict[int, tuple[str, np.ndarray]] = {}
    for path in sorted(svg_directory.rglob("*.svg")):
        stem = path.stem
        match = re.fullmatch(r"U\+([0-9A-Fa-f]{4,6})", stem)
        if 1 == len(stem) and stem.isascii() and stem.isalpha():
            codepoint = ord(stem)
        elif None is not match:
            codepoint = int(match.group(1), 16)
        else:
            problems.append(f"{path.name}: cannot tell the codepoint (name it with one ASCII letter or U+XXXX)")
            continue
        try:
            canvas = read_svg_canvas(path)
        except (ValueError, ElementTree.ParseError) as error:
            problems.append(f"{path.name}: {error}")
            continue
        if canvas_height != canvas.shape[0]:
            problems.append(f"{path.name}: canvas is {canvas.shape[0]} rows tall, Metrics.toml implies {canvas_height}")
            continue
        if left_margin + glyph_width > canvas.shape[1]:
            problems.append(f"{path.name}: canvas is {canvas.shape[1]} columns wide, narrower than the glyph box")
            continue
        box = (slice(top_margin, top_margin + glyph_height), slice(left_margin, left_margin + glyph_width))
        outside = canvas.copy()
        outside[box] = False
        if outside.any():
            cells = [f"(column {c}, row {r})" for r, c in np.argwhere(outside)[:4]]
            problems.append(f"{path.name}: {int(outside.sum())} filled cells lie outside the glyph box, e.g. {', '.join(cells)}")
            continue
        if not canvas[box].any():
            problems.append(f"{path.name}: nothing filled inside the glyph box")
            continue
        if codepoint in imported:
            problems.append(f"{path.name}: duplicates the codepoint of {imported[codepoint][0]}")
            continue
        imported[codepoint] = (path.name, canvas[box])

    matched: list[str] = []
    new_codepoints: list[int] = []
    replaced_codepoints: list[int] = []
    for codepoint, (name, grid) in sorted(imported.items()):
        existing_path = glyph_directory / f"U+{codepoint:04X}.glyph"
        if existing_path.exists():
            existing = parse_glyph(existing_path)
            rows = ["".join("#" if filled else "." for filled in row) for row in grid]
            if tuple(rows) == tuple(existing.rows):
                matched.append(chr(codepoint))
            elif overwrite:
                replaced_codepoints.append(codepoint)
            else:
                problems.append(f"{name}: differs from the existing {existing_path.name}")
                problems.append("    drawn in the SVG:  " + " ".join(rows))
                problems.append("    in the .glyph file: " + " ".join(existing.rows))
        else:
            new_codepoints.append(codepoint)

    print(f"{len(imported)} SVG glyphs read; {len(matched)} identical to the existing .glyph files ({''.join(matched)}); "
          f"{len(replaced_codepoints)} changed ({''.join(chr(codepoint) for codepoint in replaced_codepoints)}); "
          f"{len(new_codepoints)} new")
    if 0 != len(problems):
        for problem in problems:
            print(f"PROBLEM: {problem}")
        print("nothing written")
        return 1

    glyph_directory.mkdir(parents=True, exist_ok=True)
    for codepoint in sorted(new_codepoints + replaced_codepoints):
        name, grid = imported[codepoint]
        path = glyph_directory / f"U+{codepoint:04X}.glyph"
        comment = f"{chr(codepoint)}, drawn in Sketch ({name}, {glyph_width}x{glyph_height} cells at column {left_margin}, row {top_margin})"
        path.write_text(format_glyph_file(comment, glyph_width + 1, glyph_height, grid), encoding="utf-8")
        print(f"{'replaced' if codepoint in replaced_codepoints else 'wrote'} {path.name}")
        for row in grid:
            print("    " + "".join("#" if filled else "." for filled in row))
    return 0


SPECIMEN_LINES = ("BLACK SPHINX", "OF QUARTZ • ", "JUDGE MY VOW")
SPECIMEN_MARGIN = 8  # empty grid pixels on every side of the text


def make_specimen(source_directory: Path, font_path: Path, logo_path: Path, output_path: Path, pixel_size: int) -> None:
    """The one specimen image: SPECIMEN_LINES in the logo's colors, one grid pixel = `pixel_size` image pixels."""
    metrics = load_metrics(source_directory / "Metrics.toml")
    glyphs = load_glyphs(source_directory / "Glyphs")
    _, _, background, foreground = read_logo(logo_path)

    line_pitch = metrics.cap_rows + metrics.rows_above_cap + metrics.rows_below_baseline
    text_width = 0  # grid pixels from the left edge of the first glyph to the right edge of the last inked one
    for line in SPECIMEN_LINES:
        pen = 0
        for character in line:
            glyph = glyphs[ord(character)]
            if 0 != len(glyph.rows):
                text_width = max(text_width, pen + len(glyph.rows[0]))
            pen += glyph.advance
    width = SPECIMEN_MARGIN + text_width + SPECIMEN_MARGIN
    height = SPECIMEN_MARGIN + metrics.cap_rows + (len(SPECIMEN_LINES) - 1) * line_pitch + SPECIMEN_MARGIN

    font = ImageFont.truetype(str(font_path), pixel_size * metrics.units_per_em // metrics.units_per_pixel)
    image = Image.new("RGB", (width * pixel_size, height * pixel_size), background)
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(SPECIMEN_LINES):
        baseline_row = SPECIMEN_MARGIN + metrics.cap_rows + index * line_pitch
        draw.text((SPECIMEN_MARGIN * pixel_size, baseline_row * pixel_size), line, font=font, fill=foreground, anchor="ls")
    image.save(output_path)
    print(f"wrote {output_path} ({image.size[0]}x{image.size[1]} px)")


def main() -> int:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-directory", type=Path, default=script_directory)
    subcommands = parser.add_subparsers(dest="command", required=True)

    extract = subcommands.add_parser("extract-logo")
    extract.add_argument("--logo", type=Path, required=True)
    extract.add_argument("--overwrite", action="store_true")

    import_svg_parser = subcommands.add_parser("import-svg")
    import_svg_parser.add_argument("--svg-directory", type=Path, required=True)
    import_svg_parser.add_argument("--left-margin", type=int, default=2, help="empty columns left of the glyph in each canvas")
    import_svg_parser.add_argument("--glyph-width", type=int, default=LOGO_GLYPH_SIZE)
    import_svg_parser.add_argument("--overwrite", action="store_true", help="replace existing .glyph files whose pixels differ")

    build = subcommands.add_parser("build")
    build.add_argument("--output-directory", type=Path, default=script_directory / "Build")

    verify = subcommands.add_parser("verify")
    verify.add_argument("fonts", nargs="+", type=Path)
    verify.add_argument("--pixel-size", type=int, default=24)

    specimen = subcommands.add_parser("specimen")
    specimen.add_argument("--font", type=Path, required=True)
    specimen.add_argument("--logo", type=Path, required=True)
    specimen.add_argument("--output", type=Path, required=True)
    specimen.add_argument("--pixel-size", type=int, default=24)

    arguments = parser.parse_args()
    if "extract-logo" == arguments.command:
        return extract_logo(arguments.source_directory, arguments.logo, arguments.overwrite)
    if "import-svg" == arguments.command:
        return import_svg(arguments.source_directory, arguments.svg_directory, arguments.left_margin, arguments.glyph_width, arguments.overwrite)
    if "build" == arguments.command:
        build_fonts(arguments.source_directory, arguments.output_directory)
        return 0
    if "verify" == arguments.command:
        return verify_fonts(arguments.source_directory, arguments.fonts, arguments.pixel_size)
    make_specimen(arguments.source_directory, arguments.font, arguments.logo, arguments.output, arguments.pixel_size)
    return 0


if "__main__" == __name__:
    sys.exit(main())
