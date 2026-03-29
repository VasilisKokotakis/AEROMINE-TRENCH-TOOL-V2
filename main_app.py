#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEROMINE Trench Analysis Tool - Desktop Application

Copyright (c) 2026 AEROMINE
All rights reserved.

This software is licensed under the MIT License.
Removal or modification of the AEROMINE branding and copyright notice is prohibited.

Author: Vasilis Kokotakis
Repository: https://github.com/VasilisKokotakis/AEROMINE-TRENCHE-TOOL-V2
"""

import os
import shutil
import threading
from pathlib import Path
import numpy as np
import pandas as pd
from sections.io import load_las_points
from sections.processing import auto_axis, compute_sections, validate_params
from app import build_summary, create_cross_section_plots, create_summary_analysis_plots, generate_report, run_complete_analysis
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

class AeromineApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aeromine Trench Tool")
        self.geometry("760x1000")
        self.resizable(False, False)
        self.selected_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self._init_style()
        self._build_ui()

    def _init_style(self):
        self.configure(background="#f7f9fc")
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=10)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background="#f7f9fc")
        style.configure("TFrame", background="#f7f9fc")
        style.configure("TLabelframe", background="#f7f9fc", padding=10)
        style.configure("TLabelframe.Label", background="#f7f9fc", font=(default_font.actual("family"), 10, "bold"))
        style.configure("TButton", padding=(10, 6), font=(default_font.actual("family"), 10, "bold"))
        style.map("TButton", background=[("active", "#4a90e2")], foreground=[("active", "white")])
        style.configure("TEntry", padding=4)

    def _build_ui(self):
        # Header with AEROMINE watermark
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x", padx=16, pady=(10, 5))
        watermark_label = tk.Label(
            header_frame,
            text="AEROMINE",
            font=("Arial", 28, "bold"),
            fg="#4a90e2",
            bg="#f7f9fc"
        )
        watermark_label.pack()
        subtitle_label = tk.Label(
            header_frame,
            text="Trench Analysis Tool",
            font=("Arial", 11),
            fg="#666666",
            bg="#f7f9fc"
        )
        subtitle_label.pack()

        # File selection
        file_frame = ttk.LabelFrame(self, text="1. Select LAS/LAZ File")
        file_frame.pack(fill="x", padx=16, pady=10)
        ttk.Entry(file_frame, textvariable=self.selected_file, width=60, state="readonly").pack(side="left", padx=8, pady=8, expand=True, fill="x")
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side="left", padx=8, pady=8)

        # Output directory selection
        out_frame = ttk.LabelFrame(self, text="2. Select Output Directory")
        out_frame.pack(fill="x", padx=16, pady=10)
        ttk.Entry(out_frame, textvariable=self.output_dir, width=60, state="readonly").pack(side="left", padx=8, pady=8, expand=True, fill="x")
        ttk.Button(out_frame, text="Browse...", command=self.browse_output_dir).pack(side="left", padx=8, pady=8)

        # Parameters
        param_frame = ttk.LabelFrame(self, text="3. Parameters")
        param_frame.pack(fill="x", padx=16, pady=10)
        labels = [
            ("Spacing (m):", "spacing_var", 0.10),
            ("Prefilter half-width (m):", "prefilter_var", 2.0),
            ("Edge-lock (m):", "edgelock_var", 0.05),
            ("Half-width (m):", "halfwidth_var", 0.7),
            ("Right trim (m):", "righttrim_var", 0.0),
            ("Slope threshold:", "slopethr_var", 1.5),
            ("Depth min (m):", "depthmin_var", 0.0),
        ]
        self.vars = {}
        for i, (label, varname, default) in enumerate(labels):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, padx=8, pady=6, sticky="e")
            v = tk.DoubleVar(value=default)
            self.vars[varname] = v
            ttk.Entry(param_frame, textvariable=v, width=12).grid(row=i, column=1, padx=8, pady=6, sticky="w")
        # Auto-axis and clip mode
        self.autoaxis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Auto-axis (PCA)", variable=self.autoaxis_var).grid(row=0, column=2, padx=12, pady=6, sticky="w")
        ttk.Label(param_frame, text="Clip mode:").grid(row=1, column=2, padx=12, pady=6, sticky="e")
        self.clipmode_var = tk.StringVar(value="fixed")
        ttk.Combobox(param_frame, textvariable=self.clipmode_var, values=["fixed", "auto"], width=10, state="readonly").grid(row=1, column=3, padx=12, pady=6, sticky="w")

        # Run and Analyze buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        self.btn_run = ttk.Button(btn_frame, text="Run", command=self.run_processing)
        self.btn_run.pack(side="left", padx=10)
        self.btn_analyze = ttk.Button(btn_frame, text="Analyze & Create ZIP", command=self.run_analysis)
        self.btn_analyze.pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Open Output Folder", command=self.open_output_folder).pack(side="left", padx=10)

        # Status label
        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var, foreground="#4a90e2").pack(pady=(0, 4))

        # Log/output
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=16, pady=10)
        self.log_text = tk.Text(log_frame, height=12, state="disabled", bg="#1e1e1e", fg="#dcdcdc", insertbackground="white")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("LAS/LAZ files", ".las .laz")])
        if file_path:
            self.selected_file.set(file_path)
            self.log(f"Selected file: {file_path}")

    def browse_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir.set(dir_path)
            self.log(f"Output directory: {dir_path}")

    def _set_busy(self, busy: bool, status: str = ""):
        state = "disabled" if busy else "normal"
        self.btn_run.config(state=state)
        self.btn_analyze.config(state=state)
        self.status_var.set(status)

    def run_processing(self):
        file_path = self.selected_file.get()
        out_dir = self.output_dir.get()
        if not file_path or not out_dir:
            messagebox.showerror("Error", "Please select a LAS/LAZ file and output directory.")
            return
        # Gather parameters
        spacing = self.vars["spacing_var"].get()
        prefilter = self.vars["prefilter_var"].get()
        edgelock = self.vars["edgelock_var"].get()
        halfwidth = self.vars["halfwidth_var"].get()
        righttrim = self.vars["righttrim_var"].get()
        slopethr = self.vars["slopethr_var"].get()
        depthmin = self.vars["depthmin_var"].get()
        autoaxis = self.autoaxis_var.get()
        clipmode = self.clipmode_var.get()
        # Validate parameters
        errors = validate_params(
            spacing=spacing,
            prefilter_half_width=prefilter,
            edgelock=edgelock,
            half_width=halfwidth,
            right_trim=righttrim,
            slope_thr=slopethr,
            depth_min=depthmin,
            clip_mode=clipmode,
        )
        if errors:
            messagebox.showerror("Invalid parameters", "\n".join(errors))
            return

        self._set_busy(True, "Processing...")

        def _worker():
            try:
                # Prepare output run dir
                run_id = os.path.splitext(os.path.basename(file_path))[0] + "_run"
                run_dir = Path(out_dir) / run_id
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                # Copy input file
                in_path = run_dir / os.path.basename(file_path)
                shutil.copy(file_path, in_path)
                self.after(0, self.log, f"[input] {in_path.name}")
                # Load points
                self.after(0, self.status_var.set, "Loading points...")
                x, y, z = load_las_points(str(in_path))
                self.after(0, self.log, f"[loaded] points={len(x):,}")
                # Axis
                self.after(0, self.status_var.set, "Detecting axis...")
                if autoaxis:
                    start, end = auto_axis(x, y)
                    self.after(0, self.log, "[axis] auto (PCA)")
                else:
                    start = np.array([x.min(), y.min()], dtype=float)
                    end = np.array([x.max(), y.max()], dtype=float)
                    self.after(0, self.log, "[axis] bbox diagonal")
                # Compute sections
                self.after(0, self.status_var.set, "Computing sections...")
                df = compute_sections(
                    x, y, z, start, end, spacing,
                    prefilter_half_width=prefilter,
                    edge_lock_margin=edgelock,
                )
                self.after(0, self.log, f"[sections] rows={len(df):,} sections={df['section_id'].nunique()}")
                # Clip
                self.after(0, self.status_var.set, "Clipping sections...")
                if clipmode == "fixed":
                    df = df[df["dist_off"].between(-halfwidth, halfwidth - righttrim)].copy()
                    self.after(0, self.log, f"[clip:fixed] half_width={halfwidth} right_trim={righttrim}")
                else:
                    from app import auto_edges_from_section
                    out = []
                    fail = 0
                    for sid, g in df.groupby("section_id"):
                        le, re = auto_edges_from_section(g, slope_thr=slopethr)
                        if le is None or re is None or le >= re:
                            fail += 1
                            gg = g[g["dist_off"].between(-halfwidth, halfwidth)].copy()
                            out.append(gg)
                            continue
                        gg = g[(g["dist_off"] >= le) & (g["dist_off"] <= re)].copy()
                        out.append(gg)
                    df = pd.concat(out, ignore_index=True) if out else df.iloc[0:0].copy()
                    self.after(0, self.log, f"[clip:auto] slope_thr={slopethr} failed_sections={fail}")
                    if len(df) > 0:
                        self.after(0, self.log, f"[clip:auto] dist_off=[{df['dist_off'].min():.3f},{df['dist_off'].max():.3f}]")
                # Save outputs
                self.after(0, self.status_var.set, "Saving outputs...")
                full_csv = run_dir / "sections.csv"
                summary_csv = run_dir / "sections_summary.csv"
                df.to_csv(full_csv, index=False)
                summary = build_summary(df, spacing=spacing, depth_min=depthmin)
                summary.to_csv(summary_csv, index=False)
                self.after(0, self.log, f"[output] sections.csv, sections_summary.csv saved in {run_dir}")
                self.after(0, self.log, "Processing complete. You can now run analysis.")
            except Exception as e:
                self.after(0, self.log, f"[error] {e}")
                self.after(0, messagebox.showerror, "Error", str(e))
            finally:
                self.after(0, self._set_busy, False, "")

        threading.Thread(target=_worker, daemon=True).start()

    def run_analysis(self):
        out_dir = self.output_dir.get()
        file_path = self.selected_file.get()
        if not out_dir or not file_path:
            messagebox.showerror("Error", "Please select a LAS/LAZ file and output directory.")
            return
        run_id = os.path.splitext(os.path.basename(file_path))[0] + "_run"
        run_dir = Path(out_dir) / run_id
        sections_csv = run_dir / "sections.csv"
        summary_csv = run_dir / "sections_summary.csv"
        if not sections_csv.exists() or not summary_csv.exists():
            messagebox.showerror("Error", "Please run processing first.")
            return

        self._set_busy(True, "Generating analysis...")

        def _worker():
            try:
                sections_df = pd.read_csv(sections_csv)
                summary_df = pd.read_csv(summary_csv)
                self.after(0, self.status_var.set, "Creating plots...")
                zip_path = run_complete_analysis(sections_df, summary_df, run_dir)
                self.after(0, self.log, f"[analysis] Complete analysis ZIP created: {zip_path}")
                self.after(0, messagebox.showinfo, "Analysis Complete", f"Analysis ZIP created:\n{zip_path}")
            except Exception as e:
                self.after(0, self.log, f"[error] {e}")
                self.after(0, messagebox.showerror, "Error", str(e))
            finally:
                self.after(0, self._set_busy, False, "")

        threading.Thread(target=_worker, daemon=True).start()

    def open_output_folder(self):
        out_dir = self.output_dir.get()
        if not out_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return
        import subprocess
        subprocess.Popen(["xdg-open" if os.name == "posix" else "explorer", out_dir])

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

if __name__ == "__main__":
    app = AeromineApp()
    app.mainloop()
