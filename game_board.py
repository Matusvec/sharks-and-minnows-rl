"""Sharks and Minnows board with deterministic shark pursuit."""

from dataclasses import dataclass
import math
import random

import pygame


WINDOW_SIZE = (2560, 1820)
FPS = 60
TIME_SCALE = 0.5
MAX_FRAME_SECONDS = 0.05
FIELD_ASPECT_RATIO = 120 / 53.3
FIELD_WORLD_WIDTH = FIELD_ASPECT_RATIO
FIELD_WORLD_HEIGHT = 1.0
END_ZONE_WORLD_WIDTH = FIELD_WORLD_WIDTH / 12
ACTIVE_LEFT = END_ZONE_WORLD_WIDTH
ACTIVE_RIGHT = FIELD_WORLD_WIDTH - END_ZONE_WORLD_WIDTH
SHARK_SPEED = 0.5
MINNOW_MIN_SPEED = 0.2
MINNOW_MAX_SPEED = 1.0
MINNOW_MIN_HEADING = -math.pi
MINNOW_MAX_HEADING = math.pi
MINNOW_COLLISION_RADIUS = 0.014
SHARK_COLLISION_RADIUS = 0.028

BACKGROUND = (10, 17, 24)
FIELD_GREEN = (27, 94, 57)
START_GREEN = (20, 70, 48)
SAFE_GREEN = (20, 118, 82)
FIELD_LINE = (224, 238, 230)
MUTED_LINE = (146, 185, 164)
MINNOW_COLOR = (132, 220, 255)
DEAD_MINNOW_COLOR = (225, 46, 46)
SHARK_COLOR = (245, 92, 92)
SCORE_COLOR = (132, 255, 190)
TIME_COLOR = (224, 238, 230)
PLAYER_OUTLINE = (10, 17, 24)
MINNOW_COUNT = 10


@dataclass
class Minnow:
    position: pygame.Vector2
    speed: float
    heading_radians: float = 0.0
    alive: bool = True
    safe: bool = False


@dataclass
class GameState:
    minnows: list[Minnow]
    shark_position: pygame.Vector2


def create_initial_state(seed: int | None = None) -> GameState:
    random_source = random.Random(seed)
    minnow_x = END_ZONE_WORLD_WIDTH * 0.80
    fast_minnow_index = random_source.randrange(MINNOW_COUNT)
    minnows = [
        Minnow(
            position=pygame.Vector2(
                minnow_x,
                (index + 1) / (MINNOW_COUNT + 1),
            ),
            speed=(
                MINNOW_MAX_SPEED
                if index == fast_minnow_index
                else MINNOW_MIN_SPEED
            ),
        )
        for index in range(MINNOW_COUNT)
    ]
    shark_x = ACTIVE_LEFT + (ACTIVE_RIGHT - ACTIVE_LEFT) * 0.62
    return GameState(
        minnows=minnows,
        shark_position=pygame.Vector2(
            shark_x,
            random_source.uniform(
                SHARK_COLLISION_RADIUS,
                FIELD_WORLD_HEIGHT - SHARK_COLLISION_RADIUS,
            ),
        ),
    )


def is_in_protected_zone(position: pygame.Vector2) -> bool:
    return position.x <= ACTIVE_LEFT or position.x >= ACTIVE_RIGHT


def set_minnow_heading(minnow: Minnow, heading_radians: float) -> None:
    """Set a heading in any direction."""
    if not MINNOW_MIN_HEADING <= heading_radians <= MINNOW_MAX_HEADING:
        raise ValueError("Minnow heading must be between -180 and +180 degrees")
    minnow.heading_radians = heading_radians


def select_shark_target(state: GameState) -> Minnow | None:
    """Choose the nearest eligible minnow, with the center-line tiebreaker."""
    candidates: list[tuple[float, float, int, Minnow]] = []

    for index, minnow in enumerate(state.minnows):
        if not minnow.alive or minnow.safe or is_in_protected_zone(minnow.position):
            continue

        distance_squared = state.shark_position.distance_squared_to(minnow.position)
        center_distance = abs(minnow.position.y - FIELD_WORLD_HEIGHT / 2)
        candidates.append((distance_squared, center_distance, index, minnow))

    if not candidates:
        return None

    nearest_distance = min(candidate[0] for candidate in candidates)
    tied_nearest = [
        candidate
        for candidate in candidates
        if math.isclose(candidate[0], nearest_distance, rel_tol=1e-9, abs_tol=1e-12)
    ]
    return min(tied_nearest, key=lambda candidate: (candidate[1], candidate[2]))[3]


def update_shark(state: GameState, delta_seconds: float) -> None:
    """Move the shark directly toward its current target at constant speed."""
    target = select_shark_target(state)
    if target is None or delta_seconds <= 0:
        return

    offset = target.position - state.shark_position
    distance = offset.length()
    if distance == 0:
        return

    step_distance = min(SHARK_SPEED * delta_seconds, distance)
    state.shark_position += offset.normalize() * step_distance
    state.shark_position.x = max(
        ACTIVE_LEFT + SHARK_COLLISION_RADIUS,
        min(ACTIVE_RIGHT - SHARK_COLLISION_RADIUS, state.shark_position.x),
    )
    state.shark_position.y = max(
        SHARK_COLLISION_RADIUS,
        min(FIELD_WORLD_HEIGHT - SHARK_COLLISION_RADIUS, state.shark_position.y),
    )


def update_minnows(state: GameState, delta_seconds: float) -> None:
    """Move active minnows in their selected direction."""
    if delta_seconds <= 0:
        return

    for minnow in state.minnows:
        if not minnow.alive or minnow.safe:
            continue

        direction = pygame.Vector2(
            math.cos(minnow.heading_radians),
            math.sin(minnow.heading_radians),
        )
        minnow.position += direction * minnow.speed * delta_seconds
        minnow.position.x = max(
            MINNOW_COLLISION_RADIUS,
            min(ACTIVE_RIGHT, minnow.position.x),
        )
        minnow.position.y = max(
            MINNOW_COLLISION_RADIUS,
            min(FIELD_WORLD_HEIGHT - MINNOW_COLLISION_RADIUS, minnow.position.y),
        )
        if minnow.position.x >= ACTIVE_RIGHT:
            minnow.position.x = ACTIVE_RIGHT
            minnow.safe = True


def resolve_shark_collisions(state: GameState) -> None:
    """Kill each active, unprotected minnow currently touching the shark."""
    collision_distance = SHARK_COLLISION_RADIUS + MINNOW_COLLISION_RADIUS
    collision_distance_squared = collision_distance * collision_distance

    for minnow in state.minnows:
        if not minnow.alive or minnow.safe or is_in_protected_zone(minnow.position):
            continue

        if state.shark_position.distance_squared_to(minnow.position) <= collision_distance_squared:
            minnow.alive = False


def field_rect_for(window_size: tuple[int, int]) -> pygame.Rect:
    """Return the largest centered football-field rectangle for the window."""
    window_width, window_height = window_size
    margin = max(32, min(window_width, window_height) // 12)
    available_width = max(1, window_width - 2 * margin)
    available_height = max(1, window_height - 2 * margin)

    if available_width / available_height > FIELD_ASPECT_RATIO:
        field_height = available_height
        field_width = round(field_height * FIELD_ASPECT_RATIO)
    else:
        field_width = available_width
        field_height = round(field_width / FIELD_ASPECT_RATIO)

    return pygame.Rect(
        (window_width - field_width) // 2,
        (window_height - field_height) // 2,
        field_width,
        field_height,
    )


def world_to_screen(field: pygame.Rect, position: pygame.Vector2) -> tuple[int, int]:
    return (
        round(field.left + field.width * position.x / FIELD_WORLD_WIDTH),
        round(field.top + field.height * position.y / FIELD_WORLD_HEIGHT),
    )


def draw_board(
    surface: pygame.Surface,
    state: GameState,
    elapsed_seconds: float | None = None,
    episode_seconds: float | None = None,
) -> None:
    """Draw the board and current player positions."""
    surface.fill(BACKGROUND)
    field = field_rect_for(surface.get_size())
    end_zone_width = max(1, field.width // 12)

    pygame.draw.rect(surface, FIELD_GREEN, field)

    start_zone = pygame.Rect(field.left, field.top, end_zone_width, field.height)
    safe_zone = pygame.Rect(
        field.right - end_zone_width,
        field.top,
        end_zone_width,
        field.height,
    )
    pygame.draw.rect(surface, START_GREEN, start_zone)
    pygame.draw.rect(surface, SAFE_GREEN, safe_zone)

    line_width = max(2, field.width // 450)
    pygame.draw.rect(surface, FIELD_LINE, field, line_width)

    left_goal_line = field.left + end_zone_width
    right_goal_line = field.right - end_zone_width
    playing_width = right_goal_line - left_goal_line

    pygame.draw.line(
        surface,
        FIELD_LINE,
        (left_goal_line, field.top),
        (left_goal_line, field.bottom),
        line_width,
    )
    pygame.draw.line(
        surface,
        FIELD_LINE,
        (right_goal_line, field.top),
        (right_goal_line, field.bottom),
        line_width,
    )

    for section in range(1, 10):
        x = round(left_goal_line + playing_width * section / 10)
        color = FIELD_LINE if section == 5 else MUTED_LINE
        width = line_width if section == 5 else max(1, line_width // 2)
        pygame.draw.line(surface, color, (x, field.top), (x, field.bottom), width)

    hash_length = max(5, field.height // 45)
    for section in range(1, 20):
        x = round(left_goal_line + playing_width * section / 20)
        for y_fraction in (0.38, 0.62):
            y = round(field.top + field.height * y_fraction)
            pygame.draw.line(
                surface,
                MUTED_LINE,
                (x, y - hash_length),
                (x, y + hash_length),
                max(1, line_width // 2),
            )

    label_size = max(14, field.height // 22)
    label_font = pygame.font.Font(None, label_size)
    title_font = pygame.font.Font(None, max(24, field.height // 12))
    score_font = pygame.font.Font(None, max(22, field.height // 16))

    start_label = label_font.render("MINNOW START", True, FIELD_LINE)
    safe_label = label_font.render("SAFE", True, FIELD_LINE)
    title = title_font.render("SHARKS & MINNOWS", True, FIELD_LINE)
    safe_count = sum(minnow.safe for minnow in state.minnows)
    score = score_font.render(f"SAVED: {safe_count} / {MINNOW_COUNT}", True, SCORE_COLOR)

    start_label = pygame.transform.rotate(start_label, 90)
    safe_label = pygame.transform.rotate(safe_label, -90)
    surface.blit(start_label, start_label.get_rect(center=start_zone.center))
    surface.blit(safe_label, safe_label.get_rect(center=safe_zone.center))
    surface.blit(
        title,
        title.get_rect(center=(surface.get_width() // 2, max(20, field.top // 2))),
    )
    surface.blit(
        score,
        score.get_rect(midright=(field.right, max(20, field.top // 2))),
    )
    if elapsed_seconds is not None and episode_seconds is not None:
        remaining_seconds = max(0.0, episode_seconds - elapsed_seconds)
        timer = score_font.render(
            f"TIME: {remaining_seconds:04.1f}", True, TIME_COLOR
        )
        surface.blit(
            timer,
            timer.get_rect(midleft=(field.left, max(20, field.top // 2))),
        )

    minnow_radius = max(4, field.height // 75)
    for minnow in state.minnows:
        minnow_position = world_to_screen(field, minnow.position)
        minnow_color = MINNOW_COLOR if minnow.alive else DEAD_MINNOW_COLOR
        pygame.draw.circle(
            surface,
            PLAYER_OUTLINE,
            minnow_position,
            minnow_radius + 2,
        )
        pygame.draw.circle(
            surface,
            minnow_color,
            minnow_position,
            minnow_radius,
        )

    shark_position = world_to_screen(field, state.shark_position)
    shark_radius = max(8, field.height // 38)
    pygame.draw.circle(
        surface,
        PLAYER_OUTLINE,
        shark_position,
        shark_radius + 3,
    )
    pygame.draw.circle(surface, SHARK_COLOR, shark_position, shark_radius)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Sharks & Minnows")
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    clock = pygame.time.Clock()
    state = create_initial_state()
    running = True

    while running:
        frame_seconds = min(clock.tick(FPS) / 1000, MAX_FRAME_SECONDS)
        simulation_seconds = frame_seconds * TIME_SCALE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        update_minnows(state, simulation_seconds)
        update_shark(state, simulation_seconds)
        resolve_shark_collisions(state)
        draw_board(screen, state)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
