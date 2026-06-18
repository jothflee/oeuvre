#!/usr/bin/env python3
"""
Oeuvre GUI — Tkinter interface for the SHO pipeline.

Provides a target-selection window with live log output and
an embedded image preview that replaces the cv2 preview windows.
"""

import os
import io
import base64
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pipeline import run_pipeline, scan_targets, PipelineConfig
from .config import (
    workspace,
    plate_solve_settings,
    save_plate_solve_settings,
    stack_workers_setting,
    save_stack_workers_setting,
    save_starnet_dir_setting,
)
from .projection import SkyMap
from .starfield import render_starfield
from .projection import _feather_mask
from .starnet import (
    STARNET_OFFICIAL_URL,
    managed_starnet_dir,
    starnet_status,
    resolve_starnet_source,
    strip_quarantine,
    uninstall_managed_starnet,
)


# ── Colour palette (deep-space / frosted glass) ──────────────────────────────

SPACE      = '#06070e'   # canvas behind the starfield
BG         = '#0a0c14'   # widget/frame fill — nearly deep space, so it vanishes
BG_LIGHT   = '#0b0e16'   # inner surfaces (log text)
GLASS_EDGE = '#39435c'   # hairline panel edge
FG         = '#eef2ff'   # white text floating on the stars
ACCENT     = '#89b4fa'
ACCENT2    = '#a6e3a1'
WARN       = '#fab387'
ERR        = '#f38ba8'
DIM        = '#9aa4be'   # dimmed white
FONT       = ('SF Mono', 12)
FONT_TITLE = ('SF Pro Display', 18, 'bold')
FONT_SMALL = ('SF Mono', 11)
FONT_LOG   = ('SF Mono', 10)

PREVIEW_MIN_W = 800
PREVIEW_MIN_H = 600


def _to_photo(pil_img):
    """PIL image -> Tk PhotoImage via PNG bytes.

    Avoids Pillow's ImageTk, whose `_imagingtk` C extension fails to register
    with Tk on uv / python-build-standalone interpreters ("invalid command name
    PyImagingPhoto"). Tk 8.6 reads PNG natively, so this works everywhere.
    """
    buf = io.BytesIO()
    pil_img.convert('RGB').save(buf, format='PNG')
    return tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode('ascii'))


# ── TkPreview — drop-in replacement for PipelinePreview ─────────────────────

class TkPreview:
    """Implements the same API as PipelinePreview but renders into a Tk Canvas.

    Images are always scaled to fit the available viewport while
    maintaining aspect ratio.  No scrollbars — the image is centred
    in the canvas.

    Methods (called from worker thread):
        init_grid(n_panels)
        update_panel(panel_idx, image, label)
        show_full(image, label)
        finish()

    All Tk updates are dispatched to the main thread via root.after().
    """

    def __init__(self, root, canvas, placeholder_label=None):
        self.root = root
        self.canvas = canvas
        self._placeholder = placeholder_label  # hidden on first image
        self._tk_photo = None  # prevent GC
        self._canvas_image_id = None
        self._bg_pil = None       # starfield the preview feathers onto
        self._content = None      # last image shown (PIL)
        self._n_panels = 0
        self._grid_cols = 0
        self._grid_rows = 0
        self._panel_images = []  # RGB uint8 numpy arrays per cell
        self._grid_mode = True
        self.enabled = True
        self.interactive = False

    def init_grid(self, n_panels):
        self._n_panels = n_panels
        self._grid_mode = True
        if n_panels <= 1:
            self._grid_cols, self._grid_rows = 1, 1
        elif n_panels <= 2:
            self._grid_cols, self._grid_rows = 2, 1
        elif n_panels <= 4:
            self._grid_cols, self._grid_rows = 2, 2
        elif n_panels <= 6:
            self._grid_cols, self._grid_rows = 3, 2
        else:
            self._grid_cols = 3
            self._grid_rows = (n_panels + 2) // 3
        self._panel_images = [None] * n_panels

    def update_panel(self, panel_idx, image, label=""):
        if not self.enabled:
            return
        rgb8 = self._to_rgb_uint8(image)
        rgb8 = self._add_label(rgb8, label)
        self._panel_images[panel_idx] = rgb8
        self._grid_mode = True
        self._refresh_grid()

    def show_full(self, image, label=""):
        if not self.enabled:
            return
        self._grid_mode = False
        rgb8 = self._to_rgb_uint8(image)
        rgb8 = self._add_label(rgb8, label)
        self._display(rgb8)

    def finish(self):
        pass  # GUI stays open, nothing to do

    # ── internals ──

    def _refresh_grid(self):
        filled = [img for img in self._panel_images if img is not None]
        if not filled:
            return
        cell_w = max(img.shape[1] for img in filled)
        cell_h = max(img.shape[0] for img in filled)

        grid_w = cell_w * self._grid_cols
        grid_h = cell_h * self._grid_rows

        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for i, img in enumerate(self._panel_images):
            if img is None:
                continue
            r = i // self._grid_cols
            c = i % self._grid_cols
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((cell_w, cell_h), Image.LANCZOS)
            canvas[r*cell_h:(r+1)*cell_h,
                   c*cell_w:(c+1)*cell_w] = np.array(pil_img)

        self._display(canvas)

    def set_background(self, pil_img):
        """Set the starfield the preview composites onto (no opaque box)."""
        self._bg_pil = pil_img
        self.root.after(0, self._render)

    def _display(self, rgb_array):
        """Show an RGB numpy array, feathered onto the starfield (thread-safe)."""
        self._content = Image.fromarray(rgb_array)
        self.root.after(0, self._render)

    def _render(self):
        """Composite the current image (feathered edges) onto the starfield and
        present it as a single seamless canvas image — no box background."""
        if self._placeholder is not None:
            self._placeholder.place_forget()
            self._placeholder = None

        self.canvas.update_idletasks()
        vp_w = max(self.canvas.winfo_width(), PREVIEW_MIN_W)
        vp_h = max(self.canvas.winfo_height(), PREVIEW_MIN_H)

        if self._bg_pil is not None:
            base = self._bg_pil.resize((vp_w, vp_h), Image.LANCZOS).convert('RGB')
        else:
            base = Image.new('RGB', (vp_w, vp_h), SPACE)

        if self._content is not None:
            iw, ih = self._content.size
            s = min(vp_w / iw, vp_h / ih)
            nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
            content = self._content.resize((nw, nh), Image.LANCZOS).convert('RGB')
            mask = _feather_mask(nw, nh, max(8, min(nw, nh) // 8))
            base.paste(content, ((vp_w - nw) // 2, (vp_h - nh) // 2), mask)

        self._tk_photo = _to_photo(base)
        cx, cy = vp_w // 2, vp_h // 2
        if self._canvas_image_id is not None:
            self.canvas.coords(self._canvas_image_id, cx, cy)
            self.canvas.itemconfigure(self._canvas_image_id, image=self._tk_photo)
        else:
            self._canvas_image_id = self.canvas.create_image(
                cx, cy, anchor='center', image=self._tk_photo)

    @staticmethod
    def _to_rgb_uint8(img):
        """Convert float32 [0,1] RGB or uint8 to RGB uint8."""
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                if len(img.shape) == 2:
                    return np.stack([img, img, img], axis=-1)
                return img
            out = np.clip(img, 0, 1)
            out = (out * 255).astype(np.uint8)
            if len(out.shape) == 2:
                return np.stack([out, out, out], axis=-1)
            return out
        return img

    @staticmethod
    def _add_label(img, text, bar_h=28):
        """Add a text label bar on top of the image."""
        if not text:
            return img
        h, w = img.shape[:2]
        pil_img = Image.fromarray(img)
        bar = Image.new('RGB', (w, bar_h), (30, 30, 30))
        draw = ImageDraw.Draw(bar)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((8, 5), text, fill=(0, 210, 210), font=font)
        combined = Image.new('RGB', (w, h + bar_h))
        combined.paste(bar, (0, 0))
        combined.paste(pil_img, (0, bar_h))
        return np.array(combined)


# ── Main App ────────────────────────────────────────────────────────────────

class OeuvreApp:
    """Main application window."""

    def __init__(self, root, auto_target=None):
        self.root = root
        self.root.title('Oeuvre — SHO Processing Pipeline')
        self.root.configure(bg=SPACE)
        self.root.minsize(1100, 700)
        self._set_window_icon()

        # Centre on screen
        w, h = 1400, 800
        sx = (root.winfo_screenwidth() - w) // 2
        sy = (root.winfo_screenheight() - h) // 2
        root.geometry(f'{w}x{h}+{sx}+{sy}')

        self._running = False
        self._thread = None

        self._build_ui()
        self._refresh_targets()
        self._refresh_starnet_ui()

        if auto_target:
            self._auto_select(auto_target)

    def _set_window_icon(self):
        """Use the app icon for the window/title bar (and Dock where honored)."""
        try:
            p = os.path.join(os.path.dirname(__file__), 'assets', 'icon.png')
            if os.path.exists(p):
                self._icon_photo = _to_photo(
                    Image.open(p).resize((256, 256), Image.LANCZOS))
                self.root.iconphoto(True, self._icon_photo)
        except Exception:
            pass

    def _open_starnet_download(self):
        webbrowser.open(STARNET_OFFICIAL_URL, new=2, autoraise=True)
        self._log(
            f'Opened the official StarNet v2 site: {STARNET_OFFICIAL_URL}')

    def _choose_starnet(self):
        """Point Oeuvre at a StarNet folder (v2 'starnet2' or legacy 'starnet++')
        and persist the choice so it survives restarts and shows in settings."""
        path = filedialog.askdirectory(
            initialdir=workspace(),
            title='Select your StarNet folder (containing starnet2 or starnet++)',
        )
        if not path:
            return
        resolved = resolve_starnet_source(path)
        if resolved is None:
            self._log('✗ That folder has no StarNet binary '
                      '(looked for starnet2 / starnet++).')
            return
        # Clear macOS quarantine now so the first run can't stall on Gatekeeper.
        strip_quarantine(resolved, log=self._log)
        save_starnet_dir_setting(resolved)
        status = starnet_status()
        self._log(f'✓ Using StarNet: {status["binary_name"]} at {resolved}')
        self._refresh_starnet_ui()

    def _reset_starnet(self):
        """Clear the chosen folder (revert to auto-detect); optionally delete a
        leftover oeuvre-managed copy."""
        save_starnet_dir_setting('')
        dest = managed_starnet_dir()
        if os.path.isdir(dest) and messagebox.askyesno(
            'Remove managed copy',
            f'Also delete the oeuvre-managed StarNet copy at\n{dest}?',
        ):
            try:
                uninstall_managed_starnet()
                self._log(f'✓ Removed managed copy at {dest}')
            except Exception as e:
                self._log(f'✗ StarNet removal failed: {e}')
        self._log('✓ StarNet set to auto-detect')
        self._refresh_starnet_ui()

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG, font=FONT)
        style.configure('TButton', font=FONT, padding=(12, 6))
        style.configure('Title.TLabel', font=FONT_TITLE, foreground=ACCENT)
        style.configure('Dim.TLabel', font=FONT_SMALL, foreground=DIM)
        style.configure('Warn.TLabel', font=FONT_SMALL, foreground=ERR)
        style.configure('Preview.TLabel', background=BG_LIGHT)
        style.configure('Accent.TButton',
                        background=ACCENT, foreground=BG,
                        font=FONT, padding=(16, 8))
        style.map('Accent.TButton',
                  background=[('active', ACCENT2)])
        # Dark glass widgets
        style.configure('TButton', background=BG_LIGHT, foreground=FG,
                        bordercolor=GLASS_EDGE, focuscolor=GLASS_EDGE)
        style.map('TButton', background=[('active', BG)],
                  foreground=[('disabled', DIM)])
        style.configure('TCheckbutton', background=BG, foreground=FG,
                        font=FONT_SMALL)
        style.map('TCheckbutton',
                  background=[('active', BG)], foreground=[('disabled', DIM)])
        style.configure('TCombobox', fieldbackground=BG_LIGHT, background=BG,
                        foreground=FG, arrowcolor=FG, bordercolor=GLASS_EDGE)
        style.map('TCombobox', fieldbackground=[('readonly', BG_LIGHT)],
                  foreground=[('readonly', FG)])

        # ── Header ──────────────────────────────────────────────────────
        header = ttk.Frame(self.root, padding=16)
        header.pack(fill='x')
        ttk.Label(header, text='✦  Oeuvre', style='Title.TLabel').pack(
            side='left')
        ttk.Label(header, text='SHO Hubble Palette Pipeline',
                  style='Dim.TLabel').pack(side='left', padx=(12, 0))

        # ── Tab notebook ────────────────────────────────────────────────
        style.configure('TNotebook', background=SPACE, borderwidth=0)
        style.configure('TNotebook.Tab', font=FONT, padding=(16, 6),
                        background=BG_LIGHT, foreground=FG)
        style.map('TNotebook.Tab',
                  background=[('selected', BG)],
                  foreground=[('selected', ACCENT)])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=16, pady=(0, 16))

        # ── Processing tab ──────────────────────────────────────────────
        proc_tab = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(proc_tab, text='  Processing  ')
        self._build_processing_tab(proc_tab)

        # ── Settings tab ────────────────────────────────────────────────
        settings_tab = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(settings_tab, text='  Settings  ')
        self._build_settings_tab(settings_tab)

        # ── Projection tab (placeholder) ────────────────────────────────
        proj_tab = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(proj_tab, text='  Projection  ')
        self._build_projection_tab(proj_tab)

    def _build_processing_tab(self, parent):
        """Processing tab: frosted-glass cards floating over a starfield."""
        cv = tk.Canvas(parent, highlightthickness=0, bd=0, bg=SPACE)
        cv.pack(fill='both', expand=True)
        self._proc_cv = cv
        self._proc_bg_id = None
        self._proc_bg_size = None
        self._proc_bg_after = None
        self._proc_bg_photo = None

        def card():
            f = tk.Frame(cv, bg=BG, highlightbackground=GLASS_EDGE,
                         highlightcolor=GLASS_EDGE, highlightthickness=1, bd=0)
            bg = tk.Label(f, bd=0, bg=BG)            # frosted-starfield backdrop
            bg.place(relx=0, rely=0, relwidth=1, relheight=1)
            bg.lower()
            return f, bg

        # ── Controls card ────────────────────────────────────────────────
        controls, controls_bg = card()
        self._controls_card = controls
        row = tk.Frame(controls, bg=BG)
        row.pack(fill='x', padx=16, pady=(14, 8))
        ttk.Label(row, text='Target:').pack(side='left')
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(
            row, textvariable=self.target_var, state='readonly',
            font=FONT, width=24)
        self.target_combo.pack(side='left', padx=(8, 8))
        self.target_combo.bind('<<ComboboxSelected>>', self._on_target_selected)
        ttk.Button(row, text='Browse…', command=self._browse_target).pack(
            side='left', padx=(0, 8))
        ttk.Button(row, text='↻', width=3, command=self._refresh_targets).pack(
            side='left')
        # Concise filter feedback lives inline next to the controls (no long
        # path line) so the card stays compact and fits in the bundle.
        self.info_var = tk.StringVar(value='Select a target')
        ttk.Label(row, textvariable=self.info_var, style='Dim.TLabel').pack(
            side='left', padx=(14, 0))

        self.starnet_warning_var = tk.StringVar(value='')
        self.starnet_warning_label = ttk.Label(
            controls, textvariable=self.starnet_warning_var,
            style='Warn.TLabel')
        self.starnet_warning_label.pack(anchor='w', padx=16, pady=(0, 10))

        # Option vars are retained (and still wired into PipelineConfig) so they
        # can be re-exposed later; for now only "Clear cache" is shown.
        self.skip_siril_var = tk.BooleanVar(value=False)
        self.recolor_var = tk.BooleanVar(value=False)
        self.flatten_bg_var = tk.BooleanVar(value=False)
        self.truthful_var = tk.BooleanVar(value=False)
        self.clear_cache_var = tk.BooleanVar(value=False)

        opt = tk.Frame(controls, bg=BG)
        opt.pack(fill='x', padx=16, pady=(0, 6))
        ttk.Checkbutton(opt, text='Clear cache (full reprocess from raw)',
                        variable=self.clear_cache_var).pack(side='left')
        ttk.Checkbutton(opt, text='Recolour only (reuse StarNet — fast colour tweaks)',
                        variable=self.recolor_var).pack(side='left', padx=(16, 0))

        sld = tk.Frame(controls, bg=BG)
        sld.pack(fill='x', padx=16, pady=(0, 6))
        self.hue_strength_var = tk.DoubleVar(value=0.44)
        self.oiii_factor_var = tk.DoubleVar(value=0.15)
        self._make_slider(sld, 'Gold', self.hue_strength_var, 0.0, 1.0)
        self._make_slider(sld, 'Blue', self.oiii_factor_var, 0.0, 1.0,
                          padx=(24, 0))

        runrow = tk.Frame(controls, bg=BG)
        runrow.pack(fill='x', padx=16, pady=(0, 14))
        self.run_btn = ttk.Button(runrow, text='▶  Run Pipeline',
                                  style='Accent.TButton', command=self._run)
        self.run_btn.pack(side='left')
        self.status_var = tk.StringVar(value='')
        ttk.Label(runrow, textvariable=self.status_var,
                  style='Dim.TLabel').pack(side='left', padx=(16, 0))

        # ── Preview card ─────────────────────────────────────────────────
        preview_card, preview_bg = card()
        ttk.Label(preview_card, text='Preview', style='Dim.TLabel').pack(
            anchor='w', padx=12, pady=(8, 0))
        # Canvas background is the deep-space tone (not an opaque light box) so
        # the preview reads as part of the frosted starfield, never a panel.
        self.preview_canvas = tk.Canvas(preview_card, bg=SPACE, relief='flat',
                                        highlightthickness=0)
        self.preview_canvas.pack(fill='both', expand=True, padx=12, pady=(4, 12))
        self.preview_placeholder = tk.Label(
            self.preview_canvas, bg=SPACE,
            text='Pipeline preview will appear here', fg=DIM, font=FONT_SMALL)
        self.preview_placeholder.place(relx=0.5, rely=0.5, anchor='center')

        # ── Log card ─────────────────────────────────────────────────────
        log_card, log_bg = card()
        ttk.Label(log_card, text='Log', style='Dim.TLabel').pack(
            anchor='w', padx=12, pady=(8, 0))
        self.log_text = scrolledtext.ScrolledText(
            log_card, font=FONT_LOG, bg=BG_LIGHT, fg=FG, insertbackground=FG,
            relief='flat', borderwidth=0, wrap='word', state='disabled')
        self.log_text.pack(fill='both', expand=True, padx=12, pady=(4, 12))

        self.tk_preview = TkPreview(self.root, self.preview_canvas,
                                    self.preview_placeholder)

        # Embed the cards in the starfield canvas; (re)position on resize.
        self._proc_cards = {
            'controls': cv.create_window(0, 0, anchor='nw', window=controls),
            'preview': cv.create_window(0, 0, anchor='nw', window=preview_card),
            'log': cv.create_window(0, 0, anchor='nw', window=log_card),
        }
        self._card_bgs = {'controls': controls_bg, 'preview': preview_bg,
                          'log': log_bg}
        self._card_photos = {}        # keep PhotoImage refs alive
        cv.bind('<Configure>', self._relayout_proc)
        # Force a layout pass once the window is realized. In the packaged app
        # the canvas can miss its first <Configure>, leaving the preview/log
        # cards stacked at (0,0) and hidden — an explicit relayout avoids that.
        self.root.after(80, self._relayout_proc)

    def _build_settings_tab(self, parent):
        """Settings tab for StarNet setup and astrometry.net API config."""
        content = ttk.Frame(parent, padding=18)
        content.pack(fill='both', expand=True)

        def card(title):
            f = tk.Frame(content, bg=BG, highlightbackground=GLASS_EDGE,
                         highlightcolor=GLASS_EDGE, highlightthickness=1, bd=0)
            ttk.Label(f, text=title, style='Title.TLabel').pack(
                anchor='w', padx=16, pady=(12, 6))
            return f

        # StarNet setup card
        sn_card = card('StarNet v2')
        sn_card.pack(fill='x', pady=(0, 14))
        self.starnet_status_var = tk.StringVar(value='')
        ttk.Label(sn_card, textvariable=self.starnet_status_var,
                  style='Dim.TLabel').pack(anchor='w', padx=16, pady=(0, 10))
        sn_row = tk.Frame(sn_card, bg=BG)
        sn_row.pack(fill='x', padx=16)
        ttk.Button(sn_row, text='Open official download',
                   command=self._open_starnet_download).pack(side='left')
        ttk.Button(sn_row, text='Choose StarNet folder…',
                   command=self._choose_starnet).pack(side='left', padx=(8, 0))
        self.starnet_remove_btn = ttk.Button(
            sn_row, text='Reset to auto-detect', command=self._reset_starnet)
        self.starnet_remove_btn.pack(side='left', padx=(8, 0))

        ttk.Label(sn_card,
                  text='Point Oeuvre at your extracted StarNet folder — the '
                       'modern StarNet v2 ("starnet2") is preferred over the '
                       'legacy "starnet++". Your choice is remembered and shown '
                       'above.',
                  style='Dim.TLabel', wraplength=1120, justify='left').pack(
                      anchor='w', padx=16, pady=(12, 0))

        # Plate solving card
        ps_card = card('Plate Solving')
        ps_card.pack(fill='x')
        ps_info = (
            'Astrometry.net API base URL and optional API key. '
            'Local astrometry.net deployments can work without a key.')
        ttk.Label(ps_card, text=ps_info, style='Dim.TLabel', wraplength=1120,
                  justify='left').pack(anchor='w', padx=16, pady=(0, 12))

        settings = plate_solve_settings()
        self.plate_endpoint_var = tk.StringVar(value=settings['endpoint'])
        self.plate_api_key_var = tk.StringVar(value=settings['api_key'])

        form = tk.Frame(ps_card, bg=BG)
        form.pack(fill='x', padx=16)

        ttk.Label(form, text='API endpoint').grid(row=0, column=0, sticky='w')
        endpoint_entry = ttk.Entry(form, textvariable=self.plate_endpoint_var, width=64)
        endpoint_entry.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(4, 10))

        ttk.Label(form, text='API key (optional for local deployments)').grid(
            row=2, column=0, sticky='w')
        api_key_entry = ttk.Entry(form, textvariable=self.plate_api_key_var, width=64,
                                  show='*')
        api_key_entry.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(4, 10))

        form.columnconfigure(0, weight=1)

        btn_row = tk.Frame(ps_card, bg=BG)
        btn_row.pack(fill='x', padx=16, pady=(0, 10))
        ttk.Button(btn_row, text='Use public nova.astrometry.net',
                   command=self._reset_plate_solve_endpoint).pack(side='left')
        ttk.Button(btn_row, text='Save settings',
                   command=self._save_plate_solve_settings).pack(
                       side='left', padx=(8, 0))

        self.plate_status_var = tk.StringVar(value='')
        ttk.Label(ps_card, textvariable=self.plate_status_var,
                  style='Dim.TLabel').pack(anchor='w', padx=16, pady=(0, 12))

        ttk.Label(ps_card,
                  text='Use the public Astrometry.net API or point to a '
                       'compatible self-hosted endpoint. The solver reads '
                       'these values when plate solving runs.',
                  style='Dim.TLabel', wraplength=1120, justify='left').pack(
                      anchor='w', padx=16, pady=(0, 10))

        self._refresh_plate_solve_ui()

        # Processing / concurrency card
        proc_card = card('Processing')
        proc_card.pack(fill='x')
        ttk.Label(proc_card,
                  text='Number of panel/filter stacks to calibrate + register + '
                       'stack at the same time. Higher is faster on multi-core '
                       'machines but uses more memory (each stack holds all its '
                       'subs at once). 0 = automatic.',
                  style='Dim.TLabel', wraplength=1120, justify='left').pack(
                      anchor='w', padx=16, pady=(0, 12))

        self.stack_workers_var = tk.IntVar(value=stack_workers_setting())
        sw_form = tk.Frame(proc_card, bg=BG)
        sw_form.pack(fill='x', padx=16)
        ttk.Label(sw_form, text='Concurrent stack jobs (0 = auto)').pack(side='left')
        ttk.Spinbox(sw_form, from_=0, to=16, width=6,
                    textvariable=self.stack_workers_var).pack(side='left', padx=(8, 0))
        ttk.Button(sw_form, text='Save',
                   command=self._save_stack_workers).pack(side='left', padx=(8, 0))

        self.stack_workers_status_var = tk.StringVar(value='')
        ttk.Label(proc_card, textvariable=self.stack_workers_status_var,
                  style='Dim.TLabel').pack(anchor='w', padx=16, pady=(8, 12))
        self._refresh_stack_workers_ui()

    def _refresh_stack_workers_ui(self):
        if not hasattr(self, 'stack_workers_status_var'):
            return
        n = stack_workers_setting()
        self.stack_workers_status_var.set(
            'Concurrent stacking: automatic'
            if n <= 0 else f'Concurrent stacking: {n} job(s)')

    def _save_stack_workers(self):
        try:
            workers = int(self.stack_workers_var.get())
        except (tk.TclError, ValueError):
            workers = 0
        workers = max(0, min(16, workers))
        save_stack_workers_setting(workers)
        self.stack_workers_var.set(workers)
        self._refresh_stack_workers_ui()
        self._log(f'✓ Saved concurrent stack jobs: '
                  f'{"auto" if workers == 0 else workers}')

    def _refresh_starnet_ui(self):
        status = starnet_status()
        if status['available']:
            d, name = status['dir'], status['binary_name']
            if status['chosen_active']:
                text = f'Using your chosen StarNet ({name}):  {d}'
            elif status['managed_dir'] and os.path.abspath(d) == os.path.abspath(
                    status['managed_dir']):
                text = f'Installed locally ({name}) at {d}'
            else:
                text = f'Auto-detected {name} at {d}'
            warning = ''
        else:
            text = ('No StarNet found. Open the official site, download StarNet '
                    'v2, then "Choose StarNet folder…" below.')
            warning = 'Warning: StarNet is not installed. See Settings.'
        self.starnet_status_var.set(text)
        # Enable reset when there is an explicit choice or a managed copy to clear.
        if hasattr(self, 'starnet_remove_btn'):
            can_reset = bool(status['chosen_dir'] or status['managed_dir'])
            self.starnet_remove_btn.configure(
                state='normal' if can_reset else 'disabled')
        if hasattr(self, 'starnet_warning_var'):
            self.starnet_warning_var.set(warning)

    def _refresh_plate_solve_ui(self):
        settings = plate_solve_settings()
        endpoint = settings['endpoint'].strip() or 'https://nova.astrometry.net/api/'
        api_key = settings['api_key'].strip()
        if hasattr(self, 'plate_endpoint_var'):
            self.plate_endpoint_var.set(endpoint)
        if hasattr(self, 'plate_api_key_var'):
            self.plate_api_key_var.set(api_key)
        if hasattr(self, 'plate_status_var'):
            if api_key:
                self.plate_status_var.set(f'Configured endpoint: {endpoint}')
            else:
                self.plate_status_var.set(
                    f'Endpoint: {endpoint}  |  API key optional for local '
                    f'deployments')

    def _reset_plate_solve_endpoint(self):
        self.plate_endpoint_var.set('https://nova.astrometry.net/api/')
        self._refresh_plate_solve_ui()

    def _save_plate_solve_settings(self):
        endpoint = self.plate_endpoint_var.get().strip()
        api_key = self.plate_api_key_var.get().strip()
        save_plate_solve_settings(endpoint=endpoint, api_key=api_key)
        self._refresh_plate_solve_ui()
        self._log('✓ Saved plate-solving settings')

    def _proc_boxes(self, w, h):
        """Pixel boxes (x0, y0, x1, y1) for the three panels at size (w, h)."""
        pad, gap = 18, 14
        # Size the controls card to the height its content actually needs, so
        # the run button and sliders are never clipped under system fonts /
        # HiDPI in the packaged app (the old fixed 208px cut them off there).
        req = 0
        card = getattr(self, '_controls_card', None)
        if card is not None:
            try:
                req = card.winfo_reqheight()
            except Exception:
                req = 0
        ctrl_h = min(max(req + 4, 200), max(220, h // 2))
        inner_w = w - 2 * pad
        top = pad + ctrl_h + gap
        bot_h = max(140, h - top - pad)
        left_w = int(inner_w * 0.60)
        return {
            'controls': (pad, pad, pad + inner_w, pad + ctrl_h),
            'preview': (pad, top, pad + left_w, top + bot_h),
            'log': (pad + left_w + gap, top, pad + inner_w, top + bot_h),
        }

    def _set_proc_bg(self, w, h):
        """Render the starfield and frost each panel with the stars behind it."""
        if self._proc_bg_size == (w, h):
            return
        self._proc_bg_size = (w, h)
        cv = self._proc_cv
        try:
            sf = render_starfield(w, h, seed=7)
        except Exception:
            return  # degrade to the solid SPACE background

        self._proc_bg_photo = _to_photo(sf)
        if self._proc_bg_id is None:
            self._proc_bg_id = cv.create_image(0, 0, anchor='nw',
                                               image=self._proc_bg_photo)
        else:
            cv.itemconfigure(self._proc_bg_id, image=self._proc_bg_photo)
        cv.tag_lower(self._proc_bg_id)

        # Each panel background = the EXACT starfield crop behind it, so the
        # panel is seamless with the canvas — looks like no background, just the
        # border, with the stars showing straight through.
        boxes = self._proc_boxes(w, h)
        for name, lbl in self._card_bgs.items():
            x0, y0, x1, y1 = boxes[name]
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            self._card_photos[name] = _to_photo(sf.crop((x0, y0, x1, y1)))
            lbl.configure(image=self._card_photos[name])

        # Hand the preview its starfield crop; it feathers the image onto this
        # (no opaque box background).
        px0, py0, px1, py1 = boxes['preview']
        if px1 - px0 > 4 and py1 - py0 > 4:
            self.tk_preview.set_background(sf.crop((px0, py0, px1, py1)))

    def _relayout_proc(self, event=None):
        cv = self._proc_cv
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 80 or h < 80:
            return
        # Debounce the starfield regen (resize fires many events).
        if self._proc_bg_after is not None:
            cv.after_cancel(self._proc_bg_after)
        self._proc_bg_after = cv.after(140, lambda: self._set_proc_bg(w, h))

        for name, (x0, y0, x1, y1) in self._proc_boxes(w, h).items():
            cv.coords(self._proc_cards[name], x0, y0)
            cv.itemconfigure(self._proc_cards[name], width=x1 - x0, height=y1 - y0)

    def _build_projection_tab(self, parent):
        """Build Projection tab — sky map visualisation."""
        self.skymap = SkyMap()

        # ── Toolbar ─────────────────────────────────────────────────────
        toolbar = ttk.Frame(parent, padding=(16, 12, 16, 8))
        toolbar.pack(fill='x')

        ttk.Button(toolbar, text='Add FITS…',
                   command=self._proj_add_files).pack(side='left')
        ttk.Button(toolbar, text='Remove Selected',
                   command=self._proj_remove_selected).pack(
                       side='left', padx=(8, 0))
        ttk.Button(toolbar, text='Clear All',
                   command=self._proj_clear).pack(side='left', padx=(8, 0))

        sep = ttk.Separator(toolbar, orient='vertical')
        sep.pack(side='left', fill='y', padx=(16, 16), pady=2)

        ttk.Button(toolbar, text='Export PNG…',
                   command=self._proj_export).pack(side='left')

        self.proj_info_var = tk.StringVar(value='No images loaded')
        ttk.Label(toolbar, textvariable=self.proj_info_var,
                  style='Dim.TLabel').pack(side='right')

        # ── Main split: file list (left) + sky canvas (centre) + log (bottom right) ──
        content = ttk.Frame(parent, padding=(16, 0, 16, 16))
        content.pack(fill='both', expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=4)
        content.rowconfigure(0, weight=3)
        content.rowconfigure(1, weight=1)

        # File list (top-left, spans both rows)
        list_frame = ttk.Frame(content)
        list_frame.grid(row=0, column=0, rowspan=2, sticky='nsew', padx=(0, 8))
        list_frame.rowconfigure(1, weight=1)
        list_frame.columnconfigure(0, weight=1)

        ttk.Label(list_frame, text='Files',
                  style='Dim.TLabel').grid(row=0, column=0, sticky='w')

        self.proj_listbox = tk.Listbox(
            list_frame, bg=BG_LIGHT, fg=FG,
            selectbackground=ACCENT, selectforeground=BG,
            font=FONT_SMALL, relief='flat', borderwidth=0,
            selectmode='extended',
        )
        self.proj_listbox.grid(row=1, column=0, sticky='nsew', pady=(4, 0))

        list_scroll = ttk.Scrollbar(list_frame, orient='vertical',
                                     command=self.proj_listbox.yview)
        list_scroll.grid(row=1, column=1, sticky='ns', pady=(4, 0))
        self.proj_listbox.configure(yscrollcommand=list_scroll.set)

        # Sky map canvas (top-right)
        map_frame = ttk.Frame(content)
        map_frame.grid(row=0, column=1, sticky='nsew')
        map_frame.rowconfigure(1, weight=1)
        map_frame.columnconfigure(0, weight=1)

        ttk.Label(map_frame, text='Sky Map',
                  style='Dim.TLabel').grid(row=0, column=0, sticky='w')

        self.proj_canvas = tk.Canvas(
            map_frame, bg=BG_LIGHT, relief='flat',
            highlightthickness=0,
        )
        self.proj_canvas.grid(row=1, column=0, sticky='nsew', pady=(4, 0))

        proj_vscroll = ttk.Scrollbar(map_frame, orient='vertical',
                                      command=self.proj_canvas.yview)
        proj_vscroll.grid(row=1, column=1, sticky='ns', pady=(4, 0))

        proj_hscroll = ttk.Scrollbar(map_frame, orient='horizontal',
                                      command=self.proj_canvas.xview)
        proj_hscroll.grid(row=2, column=0, sticky='ew')

        self.proj_canvas.configure(xscrollcommand=proj_hscroll.set,
                                    yscrollcommand=proj_vscroll.set)

        # Log pane (bottom-right)
        log_frame = ttk.Frame(content)
        log_frame.grid(row=1, column=1, sticky='nsew', pady=(8, 0))

        ttk.Label(log_frame, text='Log',
                  style='Dim.TLabel').pack(anchor='w')

        self.proj_log = scrolledtext.ScrolledText(
            log_frame, font=FONT_LOG,
            bg=BG_LIGHT, fg=FG, insertbackground=FG,
            relief='flat', borderwidth=0,
            wrap='word', state='disabled', height=6,
        )
        self.proj_log.pack(fill='both', expand=True, pady=(4, 0))

        self.proj_canvas.configure(xscrollcommand=proj_hscroll.set,
                                    yscrollcommand=proj_vscroll.set)

        self._proj_photo = None  # prevent GC
        self._proj_canvas_id = None

        # Initial placeholder
        self.proj_canvas.update_idletasks()
        self._proj_render()

        # Re-render on resize
        self.proj_canvas.bind('<Configure>', lambda e: self._proj_render())

    # ── Projection tab actions ──────────────────────────────────────────

    def _proj_log(self, msg):
        """Append a message to the projection log panel."""
        self.proj_log.configure(state='normal')
        self.proj_log.insert('end', msg + '\n')
        self.proj_log.see('end')
        self.proj_log.configure(state='disabled')

    def _proj_add_files(self):
        paths = filedialog.askopenfilenames(
            initialdir=workspace(),
            title='Select FITS files to project',
            filetypes=[
                ('FITS files', '*.fit *.fits *.fts'),
                ('All files', '*'),
            ],
        )
        if not paths:
            return
        added = 0
        for p in paths:
            try:
                sf = self.skymap.add_file(p, log_fn=self._proj_log)
                if sf is None:
                    continue
                self.proj_listbox.insert('end', sf.label)
                self._proj_log(
                    f"✓ {sf.label}  RA={sf.ra_deg:.3f}°  "
                    f"DEC={sf.dec_deg:.3f}°  "
                    f"FOV={sf.fov_w_deg:.2f}°×{sf.fov_h_deg:.2f}°  "
                    f"{sf.width_px}×{sf.height_px}px  "
                    f"rot={sf.rotation_deg:.1f}°"
                )
                added += 1
            except Exception as e:
                import traceback
                self._proj_log(f"✗ {os.path.basename(p)}: {e}")
                self._proj_log(f"  {traceback.format_exc().splitlines()[-1]}")
        if added:
            self._proj_log(f"  Added {added} file(s), {len(self.skymap.frames)} total")
        self._proj_render()
        self._proj_update_info()

    def _proj_remove_selected(self):
        sel = list(self.proj_listbox.curselection())
        if not sel:
            return
        names = [self.proj_listbox.get(i) for i in sel]
        # Remove in reverse order to keep indices stable
        for i in reversed(sel):
            self.skymap.remove_frame(i)
            self.proj_listbox.delete(i)
        self._proj_log(f"Removed {len(names)} file(s): {', '.join(names)}")
        self._proj_render()
        self._proj_update_info()

    def _proj_clear(self):
        n = len(self.skymap.frames)
        self.skymap.frames.clear()
        self.proj_listbox.delete(0, 'end')
        self._proj_log(f"Cleared all {n} file(s)")
        self._proj_render()
        self._proj_update_info()

    def _proj_export(self):
        if not self.skymap.frames:
            self._proj_log('Export: no images loaded')
            self._proj_info('No images to export')
            return
        path = filedialog.asksaveasfilename(
            initialdir=workspace(),
            title='Export sky map as PNG',
            defaultextension='.png',
            filetypes=[('PNG', '*.png')],
            initialfile='skymap_export.png',
        )
        if not path:
            return
        try:
            self.skymap.export_png(path, width=4000, height=3000)
            self._proj_log(f"✓ Exported to {path}")
            self._proj_info(f'Exported to {os.path.basename(path)}')
        except Exception as e:
            self._proj_log(f"✗ Export failed: {e}")
            self._proj_info(f'Export failed: {e}')

    def _proj_render(self):
        """Re-render the sky map onto the canvas."""
        self.proj_canvas.update_idletasks()
        cw = max(self.proj_canvas.winfo_width(), 400)
        ch = max(self.proj_canvas.winfo_height(), 300)

        pil_img = self.skymap.render_map(canvas_w=cw, canvas_h=ch)
        self._proj_photo = _to_photo(pil_img)

        self.proj_canvas.configure(scrollregion=(0, 0, cw, ch))
        if self._proj_canvas_id is not None:
            self.proj_canvas.itemconfigure(self._proj_canvas_id,
                                           image=self._proj_photo)
        else:
            self._proj_canvas_id = self.proj_canvas.create_image(
                0, 0, anchor='nw', image=self._proj_photo)

    def _proj_update_info(self):
        n = len(self.skymap.frames)
        if n == 0:
            self.proj_info_var.set('No images loaded')
        else:
            bounds = self.skymap.get_bounds()
            span_x = bounds[2] - bounds[0]
            span_y = bounds[3] - bounds[1]
            self.proj_info_var.set(
                f"{n} frame(s)  |  FOV: {span_x:.2f}° × {span_y:.2f}°")

    def _proj_info(self, msg):
        self.proj_info_var.set(msg)

    # ── Target management ───────────────────────────────────────────────

    def _refresh_targets(self):
        self._targets = scan_targets()
        names = [name for name, _, _ in self._targets]
        self.target_combo['values'] = names
        if names and not self.target_var.get():
            self.target_combo.current(0)
            self._on_target_selected(None)

    def _browse_target(self):
        path = filedialog.askdirectory(
            initialdir=workspace(),
            title='Select target directory',
        )
        if path:
            # Accept any dir with Light/<filter>/ structure or FITS files
            light = os.path.join(path, 'Light')
            has_light = os.path.isdir(light)
            has_fits = has_light or any(
                f.lower().endswith(('.fits', '.fit'))
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            )
            if not has_fits:
                self._log("WARNING: No Light/ subdirectory or FITS files "
                          f"found in {path}\n")
                return
            name = os.path.basename(path)
            mode = 'Light/<filter>/' if has_light else 'auto-detect filters'
            self._log(f"Added target: {name} ({mode})\n")
            values = list(self.target_combo['values'])
            if name not in values:
                self._targets.append((name, path, []))
                values.append(name)
                self.target_combo['values'] = values
            self.target_var.set(name)
            self._on_target_selected(None)

    def _on_target_selected(self, _event):
        name = self.target_var.get()
        for tname, tpath, tfilters in self._targets:
            if tname == name:
                filt_str = ', '.join(tfilters) if tfilters else '(scanning…)'
                self.info_var.set(f'Filters: {filt_str}')
                return
        self.info_var.set('')

    def _auto_select(self, target):
        """Select a target by name or path (used for CLI auto-launch)."""
        target = os.path.abspath(target)
        name = os.path.basename(target)
        found = False
        for tname, tpath, _ in self._targets:
            if tname == name or tpath == target:
                self.target_var.set(tname)
                self._on_target_selected(None)
                found = True
                break
        if not found:
            if os.path.isdir(target):
                self._targets.append((name, target, []))
                values = list(self.target_combo['values'])
                values.append(name)
                self.target_combo['values'] = values
                self.target_var.set(name)
                self._on_target_selected(None)

    # ── Logging ─────────────────────────────────────────────────────────

    def _make_slider(self, parent, label, variable, from_, to,
                     padx=(0, 0), width=160):
        """Add a labelled scale slider to *parent* (packed left)."""
        cell = ttk.Frame(parent)
        cell.pack(side='left', padx=padx)

        ttk.Label(cell, text=label, style='Dim.TLabel').pack(side='left')

        val_label = ttk.Label(cell, text=f'{variable.get():.2f}',
                              style='Dim.TLabel', width=4)
        val_label.pack(side='right', padx=(4, 0))

        def _on_change(v, lbl=val_label):
            lbl.configure(text=f'{float(v):.2f}')

        scale = tk.Scale(
            cell, variable=variable,
            from_=from_, to=to, resolution=0.01,
            orient='horizontal', length=width,
            bg=BG_LIGHT, fg=FG, troughcolor=BG,
            highlightthickness=0, bd=0, sliderrelief='flat',
            activebackground=ACCENT,
            command=_on_change,
            showvalue=False,
        )
        scale.pack(side='left', padx=(6, 0))

    def _log(self, msg):
        """Append to log panel (thread-safe)."""
        def _append():
            self.log_text.configure(state='normal')
            self.log_text.insert('end', msg + '\n')
            self.log_text.see('end')
            self.log_text.configure(state='disabled')
        self.root.after(0, _append)

    # ── Pipeline execution ──────────────────────────────────────────────

    def _get_target_path(self):
        name = self.target_var.get()
        for tname, tpath, _ in self._targets:
            if tname == name:
                return tpath
        return None

    def _run(self):
        if self._running:
            return

        target_path = self._get_target_path()
        if not target_path:
            self._log("ERROR: No target selected")
            return

        self._running = True
        self.run_btn.configure(state='disabled')
        self.status_var.set('Running…')

        # Clear log
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

        # Reset preview to the transparent frosted starfield — no opaque
        # "processing" box. Progress is shown in the status line and log.
        if getattr(self, 'preview_placeholder', None) is not None:
            self.preview_placeholder.place_forget()
            self.preview_placeholder = None
        self.preview_canvas.delete('all')
        self.tk_preview._canvas_image_id = None
        self.tk_preview._content = None
        self.tk_preview._placeholder = None
        self.tk_preview._panel_images = []
        self.tk_preview._render()  # re-composite the starfield (no content yet)

        cfg = PipelineConfig(
            target=target_path,
            no_preprocess=self.skip_siril_var.get(),
            recolor_only=self.recolor_var.get(),
            clear_cache=self.clear_cache_var.get(),
            flatten_background=self.flatten_bg_var.get(),
            truthful_mode=self.truthful_var.get(),
            hue_strength=round(self.hue_strength_var.get(), 3),
            oiii_factor=round(self.oiii_factor_var.get(), 3),
            stack_workers=stack_workers_setting(),
            no_preview=True,   # disable cv2 windows
            interactive=False,
            log_callback=self._log,
            preview_object=self.tk_preview,  # inject Tk preview
        )

        self._thread = threading.Thread(
            target=self._run_worker, args=(cfg,), daemon=True,
        )
        self._thread.start()

    def _run_worker(self, cfg):
        try:
            output = run_pipeline(cfg)
            self._log(f"\n✓ Pipeline complete: {output}")
            self.root.after(0, lambda: self.status_var.set(f'Done — {output}'))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            err_msg = str(e)
            self._log(f"\n✗ ERROR: {err_msg}")
            self._log(f"\n--- Traceback ---\n{tb}")
            self.root.after(0, lambda m=err_msg: self.status_var.set(f'Error: {m}'))
        finally:
            self._running = False
            self.root.after(0, lambda: self.run_btn.configure(state='normal'))


def launch_gui(auto_target=None):
    """Create and run the Oeuvre GUI."""
    root = tk.Tk()
    app = OeuvreApp(root, auto_target=auto_target)
    root.mainloop()
