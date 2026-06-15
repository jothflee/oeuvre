#!/usr/bin/env python3
"""
Oeuvre GUI — Tkinter interface for the SHO pipeline.

Provides a target-selection window with live log output and
an embedded image preview that replaces the cv2 preview windows.
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

from .pipeline import run_pipeline, scan_targets, PipelineConfig
from .config import workspace
from .projection import SkyMap


# ── Colour palette ──────────────────────────────────────────────────────────

BG         = '#1e1e2e'
BG_LIGHT   = '#2a2a3d'
FG         = '#cdd6f4'
ACCENT     = '#89b4fa'
ACCENT2    = '#a6e3a1'
WARN       = '#fab387'
ERR        = '#f38ba8'
DIM        = '#6c7086'
FONT       = ('SF Mono', 12)
FONT_TITLE = ('SF Pro Display', 18, 'bold')
FONT_SMALL = ('SF Mono', 11)
FONT_LOG   = ('SF Mono', 10)

PREVIEW_MIN_W = 800
PREVIEW_MIN_H = 600


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

    def _display(self, rgb_array):
        """Show an RGB numpy array on the Tk canvas (thread-safe).

        Always scales the image to fit the viewport while maintaining
        aspect ratio.  The image is centred in the canvas.
        """
        pil_img = Image.fromarray(rgb_array)

        def _update(img=pil_img):
            # Hide placeholder text on first real image
            if self._placeholder is not None:
                self._placeholder.place_forget()
                self._placeholder = None

            # Measure the current viewport size
            self.canvas.update_idletasks()
            vp_w = self.canvas.winfo_width()
            vp_h = self.canvas.winfo_height()
            vp_w = max(vp_w, PREVIEW_MIN_W)
            vp_h = max(vp_h, PREVIEW_MIN_H)

            iw, ih = img.size

            # Always scale to fit viewport (up or down)
            scale = min(vp_w / iw, vp_h / ih)
            iw_new, ih_new = int(iw * scale), int(ih * scale)
            display_img = img.resize((iw_new, ih_new), Image.LANCZOS)

            self._tk_photo = ImageTk.PhotoImage(display_img)

            # Centre the image in the canvas
            cx, cy = vp_w // 2, vp_h // 2

            if self._canvas_image_id is not None:
                self.canvas.coords(self._canvas_image_id, cx, cy)
                self.canvas.itemconfigure(self._canvas_image_id,
                                          image=self._tk_photo)
            else:
                self._canvas_image_id = self.canvas.create_image(
                    cx, cy, anchor='center', image=self._tk_photo)

        self.root.after(0, _update)

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
        self.root.configure(bg=BG)
        self.root.minsize(1100, 700)

        # Centre on screen
        w, h = 1400, 800
        sx = (root.winfo_screenwidth() - w) // 2
        sy = (root.winfo_screenheight() - h) // 2
        root.geometry(f'{w}x{h}+{sx}+{sy}')

        self._running = False
        self._thread = None

        self._build_ui()
        self._refresh_targets()

        if auto_target:
            self._auto_select(auto_target)

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG, font=FONT)
        style.configure('TButton', font=FONT, padding=(12, 6))
        style.configure('Title.TLabel', font=FONT_TITLE, foreground=ACCENT)
        style.configure('Dim.TLabel', font=FONT_SMALL, foreground=DIM)
        style.configure('Preview.TLabel', background=BG_LIGHT)
        style.configure('Accent.TButton',
                        background=ACCENT, foreground=BG,
                        font=FONT, padding=(16, 8))
        style.map('Accent.TButton',
                  background=[('active', ACCENT2)])

        # ── Header ──────────────────────────────────────────────────────
        header = ttk.Frame(self.root, padding=16)
        header.pack(fill='x')
        ttk.Label(header, text='✦  Oeuvre', style='Title.TLabel').pack(
            side='left')
        ttk.Label(header, text='SHO Hubble Palette Pipeline',
                  style='Dim.TLabel').pack(side='left', padx=(12, 0))

        # ── Tab notebook ────────────────────────────────────────────────
        style.configure('TNotebook', background=BG)
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

        # ── Projection tab (placeholder) ────────────────────────────────
        proj_tab = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(proj_tab, text='  Projection  ')
        self._build_projection_tab(proj_tab)

    def _build_processing_tab(self, parent):
        """Build all Processing tab contents into *parent*."""

        # ── Target selection ────────────────────────────────────────────
        sel_frame = ttk.Frame(parent, padding=(16, 12, 16, 8))
        sel_frame.pack(fill='x')

        ttk.Label(sel_frame, text='Target:').pack(side='left')

        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(
            sel_frame, textvariable=self.target_var,
            state='readonly', font=FONT, width=30,
        )
        self.target_combo.pack(side='left', padx=(8, 8))
        self.target_combo.bind('<<ComboboxSelected>>', self._on_target_selected)

        ttk.Button(sel_frame, text='Browse…',
                   command=self._browse_target).pack(side='left', padx=(0, 8))

        ttk.Button(sel_frame, text='↻', width=3,
                   command=self._refresh_targets).pack(side='left')

        # ── Info line ───────────────────────────────────────────────────
        self.info_var = tk.StringVar(value='Select a target to begin')
        info_frame = ttk.Frame(parent, padding=(16, 0, 16, 8))
        info_frame.pack(fill='x')
        ttk.Label(info_frame, textvariable=self.info_var,
                  style='Dim.TLabel').pack(side='left')

        # ── Options row ─────────────────────────────────────────────────
        opt_frame = ttk.Frame(parent, padding=(16, 0, 16, 8))
        opt_frame.pack(fill='x')

        self.skip_siril_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Skip preprocessing (reuse masters)',
                        variable=self.skip_siril_var).pack(side='left')

        self.recolor_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Recolor only (skip to color balance)',
                        variable=self.recolor_var).pack(side='left', padx=(16, 0))

        self.clear_cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Clear cache (full reprocess)',
                        variable=self.clear_cache_var).pack(side='left', padx=(16, 0))

        self.flatten_bg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Flatten background (cloud/gradient removal)',
                        variable=self.flatten_bg_var).pack(side='left', padx=(16, 0))

        self.truthful_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text='Truthful (no star suppression, no SCNR)',
                        variable=self.truthful_var).pack(side='left', padx=(16, 0))

        # ── Color tuning sliders ─────────────────────────────────────────
        slider_frame = ttk.Frame(parent, padding=(16, 0, 16, 8))
        slider_frame.pack(fill='x')

        self.hue_strength_var = tk.DoubleVar(value=0.44)
        self.oiii_factor_var  = tk.DoubleVar(value=0.38)

        self._make_slider(slider_frame, 'Gold',
                          self.hue_strength_var, 0.0, 1.0)
        self._make_slider(slider_frame, 'Blue',
                          self.oiii_factor_var,  0.0, 1.0, padx=(24, 0))

        # ── Run button ──────────────────────────────────────────────────
        btn_frame = ttk.Frame(parent, padding=(16, 4, 16, 8))
        btn_frame.pack(fill='x')

        self.run_btn = ttk.Button(
            btn_frame, text='▶  Run Pipeline', style='Accent.TButton',
            command=self._run,
        )
        self.run_btn.pack(side='left')

        self.status_var = tk.StringVar(value='')
        ttk.Label(btn_frame, textvariable=self.status_var,
                  style='Dim.TLabel').pack(side='left', padx=(16, 0))

        # ── Main content: preview (left) + log (right) ─────────────────
        content = ttk.Frame(parent, padding=(16, 0, 16, 16))
        content.pack(fill='both', expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        # Preview pane (left) — scrollable canvas
        preview_frame = ttk.Frame(content)
        preview_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        ttk.Label(preview_frame, text='Preview',
                  style='Dim.TLabel').grid(row=0, column=0, sticky='w')

        self.preview_canvas = tk.Canvas(
            preview_frame, bg=BG_LIGHT, relief='flat',
            highlightthickness=0,
        )
        self.preview_canvas.grid(row=1, column=0, sticky='nsew', pady=(4, 0))

        # Placeholder text (hidden when first image arrives)
        self.preview_placeholder = tk.Label(
            self.preview_canvas, bg=BG_LIGHT,
            text='Pipeline preview will appear here',
            fg=DIM, font=FONT_SMALL,
        )
        self.preview_placeholder.place(relx=0.5, rely=0.5, anchor='center')

        # Log pane (right)
        log_frame = ttk.Frame(content)
        log_frame.grid(row=0, column=1, sticky='nsew')

        ttk.Label(log_frame, text='Log',
                  style='Dim.TLabel').pack(anchor='w')

        self.log_text = scrolledtext.ScrolledText(
            log_frame, font=FONT_LOG,
            bg=BG_LIGHT, fg=FG, insertbackground=FG,
            relief='flat', borderwidth=0,
            wrap='word', state='disabled',
        )
        self.log_text.pack(fill='both', expand=True, pady=(4, 0))

        # Create the TkPreview object
        self.tk_preview = TkPreview(self.root, self.preview_canvas,
                                    self.preview_placeholder)

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
        self._proj_photo = ImageTk.PhotoImage(pil_img)

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
                self.info_var.set(f'{tpath}  —  Filters: {filt_str}')
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

        # Reset preview
        self.preview_canvas.delete('all')
        self.tk_preview._canvas_image_id = None
        self.preview_placeholder = tk.Label(
            self.preview_canvas, bg=BG_LIGHT,
            text='Processing…', fg=DIM, font=FONT_SMALL,
        )
        self.preview_placeholder.place(relx=0.5, rely=0.5, anchor='center')
        self.tk_preview._placeholder = self.preview_placeholder

        cfg = PipelineConfig(
            target=target_path,
            no_preprocess=self.skip_siril_var.get(),
            recolor_only=self.recolor_var.get(),
            clear_cache=self.clear_cache_var.get(),
            flatten_background=self.flatten_bg_var.get(),
            truthful_mode=self.truthful_var.get(),
            hue_strength=round(self.hue_strength_var.get(), 3),
            oiii_factor=round(self.oiii_factor_var.get(), 3),
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
