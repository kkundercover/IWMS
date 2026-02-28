import pygame
from sys import exit
import heapq
import random
import time
from collections import defaultdict
from itertools import permutations

# ─── INIT ─────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((1500, 700))
pygame.display.set_caption("IWMS Simulation - Smart City Waste Management")
clock = pygame.time.Clock()

# ─── LOAD MAP ─────────────────────────────────────────────────────────────────
surface = pygame.image.load('map_bg2.png').convert()
MAP_WIDTH, MAP_HEIGHT = surface.get_size()

# ─── COLORS ───────────────────────────────────────────────────────────────────
GREEN        = (34,  139, 34 )
GREEN_HOVER  = (100, 220, 100)
RED          = (220, 30,  30 )
RED_HOVER    = (255, 90,  90 )
TEXT_COLOR   = (255, 255, 255)
DARK_BG      = (20,  20,  20 )
BTN_NORMAL   = (20,  160, 80 )
BTN_HOVER_C  = (40,  200, 100)
BTN_PRESS    = (10,  100, 50 )
BTN_BLUE     = (30,  100, 200)
BTN_BLUE_H   = (60,  140, 240)
BTN_BLUE_P   = (15,  60,  140)
ALERT_YELLOW = (255, 200, 0  )
ALERT_RED    = (220, 40,  40 )
BIO_COLOR    = (80,  200, 80 )    # green bar for biodegradable
DRY_COLOR    = (200, 140, 40 )    # amber bar for dry waste

SEGMENT_COLORS = [
    (255, 80,  0  ),
    (0,   160, 255),
    (220, 0,   220),
    (255, 220, 0  ),
    (0,   220, 180),
    (255, 120, 180),
    (160, 255, 80 ),
]

# ─── WAYPOINTS (area names + coordinates) ─────────────────────────────────────
AREA_NAMES = [
    "Sector 7",
    "North Hub",
    "East Zone",
    "Central",
    "West Park",
    "Old Town",
    "Tech District",
]

WAYPOINTS = [
    (750,  350),
    (520,  38 ),
    (1032, 459),
    (961,  281),
    (627,  498),
    (100,  300),
    (1400, 300),
]

# ─── POPULATION DENSITIES (fixed, assigned once) ──────────────────────────────
# Each area gets a random density between 100–1000, constant throughout execution
random.seed(42)   # fixed seed so densities never change between runs
POPULATION_DENSITY = [random.randint(100, 1000) for _ in WAYPOINTS]
random.seed()     # restore true randomness for everything else

# ─── WASTE CAPACITY FORMULA ───────────────────────────────────────────────────
# At pop density 500 → max = 2000 kg for each waste type
# Scale linearly:  max_waste = (density / 500) * 2000  =  density * 4
def max_waste(density):
    return int((density / 500) * 2000)

MAX_BIO = [max_waste(d) for d in POPULATION_DENSITY]
MAX_DRY = [max_waste(d) for d in POPULATION_DENSITY]

# ─── INITIAL WASTE STATE ──────────────────────────────────────────────────────
current_bio = [0] * len(WAYPOINTS)
current_dry = [0] * len(WAYPOINTS)

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
WHITE_THRESHOLD = 95
POINT_RADIUS    = 7
PATH_WIDTH      = 7
SNAP_RADIUS     = 30
RANDOM_COUNT    = None
MSG_DURATION    = 5.0


# ══════════════════════════════════════════════════════════════════════════════
#  ROAD MASK
# ══════════════════════════════════════════════════════════════════════════════
def build_road_mask(surf, threshold=WHITE_THRESHOLD):
    w, h = surf.get_size()
    rgb  = surf.copy().convert(24)
    pa   = pygame.PixelArray(rgb)
    mask = [[False] * h for _ in range(w)]
    for x in range(w):
        for y in range(h):
            raw = pa[x, y]
            r   = (raw >> 16) & 0xFF
            g   = (raw >>  8) & 0xFF
            b   =  raw        & 0xFF
            if r >= threshold and g >= threshold and b >= threshold:
                mask[x][y] = True
    del pa
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  SNAP TO ROAD
# ══════════════════════════════════════════════════════════════════════════════
def snap_to_road(mask, point, w, h, radius=SNAP_RADIUS):
    x, y = point
    if 0 <= x < w and 0 <= y < h and mask[x][y]:
        return point
    best, best_d = None, float('inf')
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and mask[nx][ny]:
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d, best = d, (nx, ny)
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  A*
# ══════════════════════════════════════════════════════════════════════════════
DIRS = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
COST = [1.4142,  1.0,  1.4142,  1.0,  1.0, 1.4142,  1.0, 1.4142]

def heuristic(a, b):
    dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
    return max(dx, dy) + 0.4142 * min(dx, dy)

def astar(mask, start, goal, w, h):
    if start == goal:
        return [start]
    open_heap = [(0.0, start)]
    came_from = {}
    g = defaultdict(lambda: float('inf'))
    g[start] = 0.0
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path
        cx, cy = cur
        for (dx, dy), cost in zip(DIRS, COST):
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < w and 0 <= ny < h and mask[nx][ny]:
                nb = (nx, ny)
                tg = g[cur] + cost
                if tg < g[nb]:
                    came_from[nb] = cur
                    g[nb] = tg
                    heapq.heappush(open_heap, (tg + heuristic(nb, goal), nb))
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMAL VISIT ORDER
# ══════════════════════════════════════════════════════════════════════════════
def best_visit_order(points):
    n = len(points)
    if n <= 2:
        return list(range(n))

    def dist(a, b):
        return ((points[a][0]-points[b][0])**2 + (points[a][1]-points[b][1])**2) ** 0.5

    if n <= 8:
        best_ord, best_d = None, float('inf')
        for perm in permutations(range(1, n)):
            order = [0] + list(perm)
            total = sum(dist(order[k], order[k+1]) for k in range(n-1))
            if total < best_d:
                best_d, best_ord = total, order
        return best_ord
    else:
        unvisited = list(range(n))
        order = [unvisited.pop(0)]
        while unvisited:
            last    = order[-1]
            closest = min(unvisited, key=lambda i: dist(last, i))
            order.append(closest)
            unvisited.remove(closest)
        return order


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMPUTE PATHS
# ══════════════════════════════════════════════════════════════════════════════
def recompute_paths(snapped_pts):
    if len(snapped_pts) < 2:
        return list(range(len(snapped_pts))), []
    visit_order = best_visit_order(snapped_pts)
    segments = []
    for k in range(len(visit_order) - 1):
        a   = snapped_pts[visit_order[k]]
        b   = snapped_pts[visit_order[k+1]]
        seg = astar(road_mask, a, b, MAP_WIDTH, MAP_HEIGHT)
        segments.append(seg)
    return visit_order, segments


# ══════════════════════════════════════════════════════════════════════════════
#  DRAW HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def draw_segment(surf, path, color):
    if not path or len(path) < 2:
        return
    step = max(1, len(path) // 3000)
    pts  = path[::step]
    if pts[-1] != path[-1]:
        pts.append(path[-1])
    pygame.draw.lines(surf, color, False, pts, PATH_WIDTH)


def draw_status(surf, font, msg):
    bar = pygame.Rect(0, MAP_HEIGHT - 30, MAP_WIDTH, 30)
    pygame.draw.rect(surf, DARK_BG, bar)
    surf.blit(font.render(msg, True, TEXT_COLOR), (10, MAP_HEIGHT - 22))


def draw_legend(surf, font_sm, visit_order, active_indices):
    if len(visit_order) < 2:
        return
    x, y = 10, 10
    for k in range(len(visit_order) - 1):
        color = SEGMENT_COLORS[k % len(SEGMENT_COLORS)]
        a_wp  = active_indices[visit_order[k]]
        b_wp  = active_indices[visit_order[k+1]]
        pygame.draw.rect(surf, color, (x, y, 14, 14))
        surf.blit(font_sm.render(f"  {AREA_NAMES[a_wp]} -> {AREA_NAMES[b_wp]}",
                                 True, TEXT_COLOR, DARK_BG), (x + 18, y))
        y += 18


def draw_button(surf, font, rect, label, mouse_pos, pressed,
                nc=BTN_NORMAL, hc=BTN_HOVER_C, pc=BTN_PRESS):
    if pressed:
        color = pc
    elif rect.collidepoint(mouse_pos):
        color = hc
    else:
        color = nc
    pygame.draw.rect(surf, color, rect, border_radius=6)
    pygame.draw.rect(surf, TEXT_COLOR, rect, 2, border_radius=6)
    txt = font.render(label, True, TEXT_COLOR)
    surf.blit(txt, txt.get_rect(center=rect.center))


def draw_alert_symbol(surf, cx, cy, radius=10):
    """Draw a yellow triangle with '!' warning symbol."""
    pts = [
        (cx,          cy - radius),
        (cx + radius, cy + radius),
        (cx - radius, cy + radius),
    ]
    pygame.draw.polygon(surf, ALERT_YELLOW, pts)
    pygame.draw.polygon(surf, (0, 0, 0), pts, 2)
    # exclamation mark
    pygame.draw.line(surf, (0, 0, 0), (cx, cy - 4), (cx, cy + 4), 2)
    pygame.draw.circle(surf, (0, 0, 0), (cx, cy + 7), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS PANEL  (bar chart overlay)
# ══════════════════════════════════════════════════════════════════════════════
PANEL_W = 900
PANEL_H = 520
PANEL_X = (1500 - PANEL_W) // 2
PANEL_Y = (700  - PANEL_H) // 2

def draw_status_panel(surf, fonts):
    font_sm, font_md, font_big = fonts

    # Dark translucent background
    panel = pygame.Surface((PANEL_W, PANEL_H), pygame.SRCALPHA)
    panel.fill((10, 15, 25, 235))
    surf.blit(panel, (PANEL_X, PANEL_Y))
    pygame.draw.rect(surf, (80, 160, 255), (PANEL_X, PANEL_Y, PANEL_W, PANEL_H), 2, border_radius=10)

    # Title
    title = font_big.render("Waste Collection Status", True, (80, 200, 255))
    surf.blit(title, (PANEL_X + (PANEL_W - title.get_width()) // 2, PANEL_Y + 12))

    # Column headers
    hdr_y = PANEL_Y + 55
    surf.blit(font_sm.render("Area", True, (180, 180, 180)),         (PANEL_X + 20,  hdr_y))
    surf.blit(font_sm.render("Pop.", True, (180, 180, 180)),         (PANEL_X + 145, hdr_y))
    surf.blit(font_sm.render("Biodegradable (kg)", True, (80,200,80)),  (PANEL_X + 205, hdr_y))
    surf.blit(font_sm.render("Dry Waste (kg)", True, (200,160,40)),  (PANEL_X + 500, hdr_y))
    surf.blit(font_sm.render("Status", True, (180,180,180)),         (PANEL_X + 790, hdr_y))

    BAR_MAX_W = 240
    row_h     = 52
    start_y   = hdr_y + 22

    for i, name in enumerate(AREA_NAMES):
        ry      = start_y + i * row_h
        density = POPULATION_DENSITY[i]
        bio     = current_bio[i]
        dry     = current_dry[i]
        mb      = MAX_BIO[i]
        md      = MAX_DRY[i]
        over_bio = bio > mb
        over_dry = dry > md

        # Row background
        row_bg = pygame.Surface((PANEL_W - 20, row_h - 4), pygame.SRCALPHA)
        row_bg.fill((255,255,255,12) if i % 2 == 0 else (0,0,0,0))
        surf.blit(row_bg, (PANEL_X + 10, ry))

        # Area name + density
        name_col = ALERT_RED if (over_bio or over_dry) else TEXT_COLOR
        surf.blit(font_sm.render(name, True, name_col),           (PANEL_X + 20,  ry + 6))
        surf.blit(font_sm.render(str(density), True, (160,160,160)), (PANEL_X + 145, ry + 6))

        # ── Biodegradable bar ─────────────────────────────────────────────────
        bx = PANEL_X + 205
        pygame.draw.rect(surf, (30, 50, 30), (bx, ry + 8, BAR_MAX_W, 16), border_radius=4)
        bio_w = min(int(BAR_MAX_W * bio / mb), BAR_MAX_W) if mb > 0 else 0
        bio_bar_col = ALERT_RED if over_bio else BIO_COLOR
        if bio_w > 0:
            pygame.draw.rect(surf, bio_bar_col, (bx, ry + 8, bio_w, 16), border_radius=4)
        pygame.draw.rect(surf, (60, 100, 60), (bx, ry + 8, BAR_MAX_W, 16), 1, border_radius=4)
        bio_lbl = f"{bio}/{mb}"
        surf.blit(font_sm.render(bio_lbl, True, TEXT_COLOR), (bx + BAR_MAX_W + 5, ry + 6))

        # ── Dry waste bar ─────────────────────────────────────────────────────
        dx = PANEL_X + 500
        pygame.draw.rect(surf, (50, 35, 10), (dx, ry + 8, BAR_MAX_W, 16), border_radius=4)
        dry_w = min(int(BAR_MAX_W * dry / md), BAR_MAX_W) if md > 0 else 0
        dry_bar_col = ALERT_RED if over_dry else DRY_COLOR
        if dry_w > 0:
            pygame.draw.rect(surf, dry_bar_col, (dx, ry + 8, dry_w, 16), border_radius=4)
        pygame.draw.rect(surf, (100, 70, 20), (dx, ry + 8, BAR_MAX_W, 16), 1, border_radius=4)
        dry_lbl = f"{dry}/{md}"
        surf.blit(font_sm.render(dry_lbl, True, TEXT_COLOR), (dx + BAR_MAX_W + 5, ry + 6))

        # ── Status column ─────────────────────────────────────────────────────
        sx = PANEL_X + 795
        if over_bio or over_dry:
            warn_parts = []
            if over_bio: warn_parts.append("BIO")
            if over_dry: warn_parts.append("DRY")
            pygame.draw.polygon(surf, ALERT_YELLOW,
                                [(sx+8, ry+2), (sx+18, ry+20), (sx-2, ry+20)])
            surf.blit(font_sm.render("!" , True, (0,0,0)), (sx+5, ry+5))
            surf.blit(font_sm.render("OVER: " + "+".join(warn_parts), True, ALERT_RED),
                      (sx+22, ry+6))
        else:
            surf.blit(font_sm.render("OK", True, (80, 220, 80)), (sx + 5, ry + 6))

    # Close hint
    hint = font_sm.render("Press  S  or click STATUS to close", True, (120, 120, 120))
    surf.blit(hint, (PANEL_X + (PANEL_W - hint.get_width()) // 2, PANEL_Y + PANEL_H - 24))


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════
print("Building road mask ... (may take a few seconds)")
road_mask = build_road_mask(surface)
print("Road mask ready.")

snapped_all = []
for pt in WAYPOINTS:
    s = snap_to_road(road_mask, pt, MAP_WIDTH, MAP_HEIGHT)
    snapped_all.append(s)
    if s is None:
        print(f"  WARNING: no road found near {pt}")
    elif s != pt:
        print(f"  Snapped {pt} -> {s}")

font_sm  = pygame.font.SysFont("monospace", 13)
font_md  = pygame.font.SysFont("monospace", 15, bold=True)
font_btn = pygame.font.SysFont("monospace", 18, bold=True)
font_big = pygame.font.SysFont("monospace", 22, bold=True)

# Buttons
BTN_UPDATE = pygame.Rect(1500 - 180, 10, 165, 44)
BTN_STATUS = pygame.Rect(1500 - 180, 62, 165, 44)

# Pre-bake "No dustbins" overlay
_nd_txt   = font_big.render("No filled dustbins yet!", True, (255, 220, 0))
ND_BW     = _nd_txt.get_width()  + 60
ND_BH     = _nd_txt.get_height() + 24
ND_BANNER = pygame.Surface((ND_BW, ND_BH), pygame.SRCALPHA)
ND_BANNER.fill((20, 20, 20, 200))
ND_TXT    = _nd_txt
ND_BX     = (1500 - ND_BW) // 2
ND_BY     = (700  - ND_BH) // 2

# ─── Runtime state ────────────────────────────────────────────────────────────
state = {
    "active_indices"  : [],
    "segments"        : [],
    "visit_order"     : [],
    "hovered"         : None,
    "btn_update_pressed": False,
    "btn_status_pressed": False,
    "show_panel"      : False,
    "status"          : "Press UPDATE to simulate waste collection and route dustbin truck.",
    "no_dustbin_start": None,
}


# ══════════════════════════════════════════════════════════════════════════════
#  DO UPDATE
# ══════════════════════════════════════════════════════════════════════════════
def do_update():
    global current_bio, current_dry

    valid = [i for i, s in enumerate(snapped_all) if s is not None]
    if len(valid) < 2:
        state["status"] = "Not enough road-adjacent waypoints to route."
        return

    # Assign new random waste amounts to ALL areas every update
    for i in range(len(WAYPOINTS)):
        current_bio[i] = random.randint(100, 1000)
        current_dry[i] = random.randint(100, 1000)

    # 25% chance — no dustbins are full (no routing needed)
    if random.random() < 0.25:
        state["active_indices"]    = []
        state["segments"]          = []
        state["visit_order"]       = []
        state["status"]            = ""
        state["no_dustbin_start"]  = time.monotonic()
        return

    state["no_dustbin_start"] = None

    # Select which areas to collect from this run
    count = RANDOM_COUNT if RANDOM_COUNT else random.randint(2, len(valid))
    count = min(count, len(valid))
    state["active_indices"] = random.sample(valid, count)

    snapped_active = [snapped_all[i] for i in state["active_indices"]]

    screen.blit(surface, (0, 0))
    draw_status(screen, font_md, f"Dispatching truck to {count} areas — computing route ...")
    pygame.display.flip()

    visit_order, segments = recompute_paths(snapped_active)
    state["visit_order"] = visit_order
    state["segments"]    = segments

    total_px  = sum(len(s) for s in segments if s)
    route_str = " -> ".join(AREA_NAMES[state["active_indices"][o]] for o in visit_order)

    # Count overloaded areas
    overloaded = sum(
        1 for i in state["active_indices"]
        if current_bio[i] > MAX_BIO[i] or current_dry[i] > MAX_DRY[i]
    )
    alert_str = f"  |  {overloaded} ALERT(s)" if overloaded else ""

    state["status"] = (f"Truck visiting {count} areas | Route: {route_str}"
                       f"{alert_str} | Press UPDATE to re-simulate.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
while True:
    mx, my = pygame.mouse.get_pos()

    # Hover detection
    state["hovered"] = None
    for i, pt in enumerate(WAYPOINTS):
        if abs(mx - pt[0]) <= POINT_RADIUS + 8 and \
           abs(my - pt[1]) <= POINT_RADIUS + 8:
            state["hovered"] = i
            break

    # ── Events ────────────────────────────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                do_update()
            elif event.key == pygame.K_s:
                state["show_panel"] = not state["show_panel"]

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if BTN_UPDATE.collidepoint(mx, my):
                state["btn_update_pressed"] = True
            if BTN_STATUS.collidepoint(mx, my):
                state["btn_status_pressed"] = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if state["btn_update_pressed"] and BTN_UPDATE.collidepoint(mx, my):
                do_update()
            if state["btn_status_pressed"] and BTN_STATUS.collidepoint(mx, my):
                state["show_panel"] = not state["show_panel"]
            state["btn_update_pressed"] = False
            state["btn_status_pressed"] = False

    # ── Draw ──────────────────────────────────────────────────────────────────
    screen.blit(surface, (0, 0))

    # 1. Path segments
    for k, seg in enumerate(state["segments"]):
        draw_segment(screen, seg, SEGMENT_COLORS[k % len(SEGMENT_COLORS)])

    # 2. Segment midpoint labels
    for k, seg in enumerate(state["segments"]):
        if seg and len(seg) > 1:
            mid   = seg[len(seg) // 2]
            color = SEGMENT_COLORS[k % len(SEGMENT_COLORS)]
            lbl   = font_sm.render(str(k + 1), True, TEXT_COLOR, color)
            screen.blit(lbl, (mid[0] - 5, mid[1] - 8))

    # 3. Waypoint dots + alert symbols
    for i, pt in enumerate(WAYPOINTS):
        in_route   = i in state["active_indices"]
        is_hovered = (i == state["hovered"])
        over_limit = current_bio[i] > MAX_BIO[i] or current_dry[i] > MAX_DRY[i]

        if in_route:
            pygame.draw.circle(screen, TEXT_COLOR, pt, POINT_RADIUS + 5, 2)
            dot_col = RED_HOVER if is_hovered else RED
        else:
            dot_col = GREEN_HOVER if is_hovered else GREEN

        pygame.draw.circle(screen, dot_col, pt, POINT_RADIUS)

        # Alert symbol above the dot if over capacity
        if over_limit:
            draw_alert_symbol(screen, pt[0], pt[1] - POINT_RADIUS - 14)

        # Label
        if in_route:
            local_idx = state["active_indices"].index(i)
            step_num  = (state["visit_order"].index(local_idx) + 1
                         if local_idx in state["visit_order"] else "?")
            lbl_text  = f"{AREA_NAMES[i]}(#{step_num})"
        else:
            lbl_text  = AREA_NAMES[i]

        lbl = font_sm.render(lbl_text, True, TEXT_COLOR, DARK_BG)
        screen.blit(lbl, (pt[0] + 9, pt[1] - 14))

        # Mini waste summary below label (always visible)
        summary = font_sm.render(
            f"B:{current_bio[i]}  D:{current_dry[i]}",
            True,
            ALERT_RED if over_limit else (180, 180, 180),
            DARK_BG
        )
        screen.blit(summary, (pt[0] + 9, pt[1] + 2))

    # 4. Legend
    draw_legend(screen, font_sm, state["visit_order"], state["active_indices"])

    # 5. Buttons (top-right)
    draw_button(screen, font_btn, BTN_UPDATE, "  UPDATE",
                (mx, my), state["btn_update_pressed"])
    draw_button(screen, font_btn, BTN_STATUS, "  STATUS",
                (mx, my), state["btn_status_pressed"],
                nc=BTN_BLUE, hc=BTN_BLUE_H, pc=BTN_BLUE_P)

    # 6. Status bar
    draw_status(screen, font_md, state["status"])

    # 7. "No filled dustbins yet!" overlay
    if state["no_dustbin_start"] is not None:
        elapsed   = time.monotonic() - state["no_dustbin_start"]
        remaining = MSG_DURATION - elapsed
        if remaining > 0:
            screen.blit(ND_BANNER, (ND_BX, ND_BY))
            screen.blit(ND_TXT,    (ND_BX + 30, ND_BY + 12))
            bar_frac   = remaining / MSG_DURATION
            bar_full_w = ND_BW - 20
            bar_y      = ND_BY + ND_BH + 6
            pygame.draw.rect(screen, (60, 60, 60),
                             (ND_BX + 10, bar_y, bar_full_w, 10), border_radius=5)
            pygame.draw.rect(screen, (255, 220, 0),
                             (ND_BX + 10, bar_y, int(bar_full_w * bar_frac), 10), border_radius=5)
        else:
            state["no_dustbin_start"] = None

    # 8. Status panel (drawn last so it's on top of everything)
    if state["show_panel"]:
        draw_status_panel(screen, (font_sm, font_md, font_big))

    pygame.display.update()
    clock.tick(60)