import json
import random
import pygame
import sys
import numpy as np
import math
import torch

from rae_faces import *

LATENT_DIM = 100

device = "cpu"
model = RAE(LATENT_DIM)
state = torch.load("../rae/model_faces_d100.pth", map_location=device)
model.load_state_dict(state)  # oder state, falls nur Gewichte

with open("wichtigste_latents_rae.txt", "r") as f:
    table = [int(el) for el in f.read().split(",")]

def uniform01_to_standard_normal(u: torch.Tensor) -> torch.Tensor:
    # Vermeidet +-Inf an den Rändern
    eps = torch.finfo(u.dtype).eps
    u = u.clamp(min=eps, max=1.0 - eps)
    # Φ^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    return torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)

def standard_normal_to_uniform01(sn):
    return (torch.erf(sn / math.sqrt(2.0)) + 1) / 2

def gen_image_surface(model, sliders):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # 1) Slider -> Tensor auf richtiges Device/ dtype

        z = [0.5] * 100
        for i, s in enumerate(sliders):
            z[table[i]] = s.value

        z = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        # 2) Uniform -> Standardnormal
        z = uniform01_to_standard_normal(z)

        output = model.decoder(z).detach().squeeze().cpu().numpy().transpose(1,2,0)

    # Debug: Werte prüfen (wenn min≈max≈0 → Schwarz)
    # print("min/max:", float(output.min()), float(output.max()))

    # 4) Zu uint8 + contiguous für Pygame
    int_output = np.ascontiguousarray((output * 255.0).clip(0, 255).astype(np.uint8))  # [H,W,3]
    h, w = int_output.shape[:2]

    # 5a) Variante A: frombuffer (schnell)
    surface = pygame.image.frombuffer(int_output.tobytes(), (w, h), "RGB")
    scaled_surface = pygame.transform.smoothscale(surface, (256,256))

    return scaled_surface

# --- Config ---
WIN_W, WIN_H = 1200, 800
BG = (25, 27, 35)
PANEL_BG = (18, 19, 26)
LEFT_BG = (30, 32, 42)
TEXT = (230, 232, 239)
ACCENT = (90, 170, 255)
TRACK = (60, 62, 74)
TRACK_HL = (72, 75, 90)
HANDLE = (240, 240, 245)
GRID1 = (210, 210, 220)
GRID2 = (150, 150, 160)

LEFT_PAD = 28
SLIDER_H = 26
SLIDER_GAP = 6
SLIDER_TRACK_H = 8
HANDLE_W = 10
SCROLL_SPEED = 40
FONT_SIZE = 14

pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("VaeVi (VAE-images and latent dimension meaning visualizer)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", FONT_SIZE)

# Layout rects
left_rect = pygame.Rect(0, 0, WIN_W // 2, WIN_H)
right_rect = pygame.Rect(WIN_W // 2, 0, WIN_W // 2, WIN_H)

# --- Slider model ---
class Slider:
    def __init__(self, idx, rect, value=0):
        self.idx = idx
        self.rect = pygame.Rect(rect)  # full row rect (x,y,w,h)
        self.value = value  # 0..1
        self.dragging = False

    @property
    def track_rect(self):
        x, y, w, h = self.rect
        ty = y + (h - SLIDER_TRACK_H) // 2
        return pygame.Rect(x + 80, ty, w - 80 - 60, SLIDER_TRACK_H)

    @property
    def handle_rect(self):
        t = self.track_rect
        hx = t.x + int(self.value * t.w) - HANDLE_W // 2
        return pygame.Rect(hx, t.centery - 9, HANDLE_W, 18)

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(mouse_pos) or self.track_rect.collidepoint(mouse_pos):
                self.dragging = True
                self.update_from_mouse(mouse_pos)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.update_from_mouse(mouse_pos)
            return True
        return False

    def update_from_mouse(self, pos):
        t = self.track_rect
        x = max(t.x, min(pos[0], t.right))
        self.value = (x - t.x) / max(1, t.w)

    def draw(self, surf):
        # label
        label = font.render(f"S{self.idx:03d}", True, TEXT)
        surf.blit(label, (self.rect.x + 12, self.rect.y + (self.rect.h - label.get_height()) // 2))
        # track
        track = self.track_rect
        pygame.draw.rect(surf, TRACK, track, border_radius=4)
        filled = track.copy(); filled.w = int(track.w * self.value)
        pygame.draw.rect(surf, TRACK_HL, filled, border_radius=4)
        # handle
        pygame.draw.rect(surf, HANDLE, self.handle_rect, border_radius=3)
        # Wertstext rechts vom Track
        val_text = font.render(f"{self.value:.2f}", True, TEXT)
        surf.blit(val_text, (track.right + 10, self.rect.y + (self.rect.h - val_text.get_height()) // 2))

# --- Build slider list ---
content_pad = 16
row_h = SLIDER_H + SLIDER_GAP
content_h = content_pad * 2 + LATENT_DIM * row_h
content_w = right_rect.w

sliders = []
for i in range(LATENT_DIM):
    y = content_pad + i * row_h
    sliders.append(Slider(i + 1, (content_pad, y, content_w - content_pad * 2, SLIDER_H), value=(i % 100) / 100))

scroll_y = 0
max_scroll = max(0, content_h - right_rect.h)

# --- Helper: clamp scroll ---
def set_scroll(delta):
    global scroll_y
    scroll_y = max(0, min(scroll_y + delta, max_scroll))


example_img = pygame.Surface((256, 256))

with open("code_img", "r") as f:
    code = standard_normal_to_uniform01(
        torch.tensor(
            [float(p) for p in f.read().split(",")]
        )
    )
    np_code = code.numpy()

#np_code = torch.tensor([random.random() for _ in range(LATENT_DIM)])
for slider, value in zip(sliders, np_code):
    slider.value = value

# --- Main Loop ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()
            if event.key == pygame.K_HOME:
                scroll_y = 0
            if event.key == pygame.K_END:
                scroll_y = max_scroll
            if event.key == pygame.K_RETURN:
                pass
        if event.type == pygame.MOUSEWHEEL:
            set_scroll(-event.y * SCROLL_SPEED)

        # Delegate events to sliders only if mouse is over the right panel
        if right_rect.collidepoint(pygame.mouse.get_pos()):
            # Convert mouse to content-space (add scroll)
            mx, my = pygame.mouse.get_pos()
            content_mouse = (mx - right_rect.x, my + scroll_y)
            for s in sliders:
                if s.handle_event(event, content_mouse):
                    break

    # --- Draw ---
    screen.fill(BG)

    # Left half
    pygame.draw.rect(screen, LEFT_BG, left_rect)
    # Title
    title = font.render("Generated image", True, TEXT)
    screen.blit(title, (LEFT_PAD, LEFT_PAD))

    surface = gen_image_surface(model, sliders)
    img_rect = example_img.get_rect()
    img_rect.center = (left_rect.x + left_rect.w // 2, left_rect.y + left_rect.h // 2)
    screen.blit(surface, img_rect)

    """# Center image within left half
    img_rect = example_img.get_rect()
    img_rect.center = (left_rect.x + left_rect.w // 2, left_rect.y + left_rect.h // 2)
    # Optional subtle drop shadow
    shadow = img_rect.move(3, 3)
    pygame.draw.rect(screen, (0, 0, 0), shadow, border_radius=6)
    screen.blit(example_img, img_rect)
    pygame.draw.rect(screen, (255, 255, 255), img_rect, 2, border_radius=6)
    """
    # Right half (sliders)
    pygame.draw.rect(screen, PANEL_BG, right_rect)

    # Header bar
    header_h = 0
    header_rect = pygame.Rect(right_rect.x, right_rect.y, right_rect.w, header_h)
    pygame.draw.rect(screen, (35, 37, 48), header_rect)
    hdr = font.render("128 Slider (Scroll mit Mausrad, Home/End)", True, TEXT)
    screen.blit(hdr, (right_rect.x + 14, right_rect.y + (header_h - hdr.get_height()) // 2))

    # Content surface to allow scrolling
    content_surf = pygame.Surface((content_w, content_h), pygame.SRCALPHA)

    for s in sliders:
        s.draw(content_surf)

    # Clip to panel area below header
    view_rect = pygame.Rect(right_rect.x, right_rect.y + header_h, right_rect.w, right_rect.h - header_h)
    screen.set_clip(view_rect)
    screen.blit(content_surf, (right_rect.x, right_rect.y - scroll_y + header_h))
    screen.set_clip(None)

    # Scrollbar indicator
    if max_scroll > 0:
        bar_rect = pygame.Rect(right_rect.right - 8, header_h + 8, 4, right_rect.h - header_h - 16)
        pygame.draw.rect(screen, (55, 57, 68), bar_rect, border_radius=2)
        knob_h = max(32, int(bar_rect.h * (right_rect.h / (content_h))))
        knob_y = int(bar_rect.y + (bar_rect.h - knob_h) * (scroll_y / max_scroll))
        knob_rect = pygame.Rect(bar_rect.x, knob_y, bar_rect.w, knob_h)
        pygame.draw.rect(screen, ACCENT, knob_rect, border_radius=2)

    # Divider line between halves
    pygame.draw.line(screen, (50, 52, 65), (left_rect.right, 0), (left_rect.right, WIN_H))

    pygame.display.flip()
    clock.tick(60)