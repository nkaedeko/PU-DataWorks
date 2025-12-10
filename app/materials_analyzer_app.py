"""
Materials Testing Analyzer - Complete GUI Application
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from pathlib import Path
import threading
from scipy import stats


class ParameterDialog:
    """Parameter input dialog for tensile testing"""

    def __init__(self, parent, title):
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("300x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))

        # Create widgets
        ttk.Label(self.dialog, text="Gauge Length (mm):").pack(pady=5)
        self.gauge_length = tk.StringVar(value="30.0")
        ttk.Entry(self.dialog, textvariable=self.gauge_length).pack(pady=5)

        ttk.Label(self.dialog, text="Cross-sectional Area (mm²):").pack(pady=5)
        self.cross_area = tk.StringVar(value="3.0")
        ttk.Entry(self.dialog, textvariable=self.cross_area).pack(pady=5)

        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side='left', padx=5)

    def ok_clicked(self):
        try:
            self.result = {
                'gauge_length': float(self.gauge_length.get()),
                'cross_section_area': float(self.cross_area.get())
            }
            self.dialog.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")

    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()


class MaterialsAnalyzerApp:
    """Complete Materials Testing Analyzer GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Materials Testing Analyzer v1.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Data storage
        self.tensile_data = {}
        self.tga_data = {}

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI interface"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        self.create_home_tab()
        self.create_tensile_tab()
        self.create_tga_tab()
        self.create_results_tab()

    def create_home_tab(self):
        """Create home/welcome tab"""
        home_frame = ttk.Frame(self.notebook)
        self.notebook.add(home_frame, text="Home")

        # Title
        title_label = tk.Label(home_frame, text="Materials Testing Analyzer",
                               font=('Arial', 24, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=30)

        # Description
        desc_text = """
        Welcome to the Materials Testing Analyzer!

        This application provides comprehensive analysis for:

        • Tensile Testing Data
          - Stress-strain curve analysis
          - Statistical analysis of multiple trials
          - Young's modulus, UTS, toughness calculations
          - Publication-ready plots and tables

        • TGA (Thermogravimetric Analysis) Data  
          - Thermal decomposition analysis
          - T5, T50, Tmax calculations
          - Weight loss and derivative curves
          - Multi-sample comparison

        Features:
        ✓ Easy file loading with intuitive interface
        ✓ Interactive parameter input
        ✓ Real-time data visualization
        ✓ Professional publication-ready outputs
        ✓ Excel export functionality

        Select a tab above to begin your analysis!
        """

        desc_label = tk.Label(home_frame, text=desc_text, font=('Arial', 12),
                              bg='#f0f0f0', fg='#34495e', justify='left')
        desc_label.pack(pady=20, padx=40)

        # Quick start buttons
        button_frame = ttk.Frame(home_frame)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Start Tensile Analysis",
                   command=lambda: self.notebook.select(1), width=20).pack(side='left', padx=10)
        ttk.Button(button_frame, text="Start TGA Analysis",
                   command=lambda: self.notebook.select(2), width=20).pack(side='left', padx=10)

        # Version info
        version_label = tk.Label(home_frame, text="Version 1.0 | Developed for Materials Science Research",
                                 font=('Arial', 10, 'italic'), bg='#f0f0f0', fg='#7f8c8d')
        version_label.pack(side='bottom', pady=10)

    def create_tensile_tab(self):
        """Create tensile testing analysis tab"""
        tensile_frame = ttk.Frame(self.notebook)
        self.notebook.add(tensile_frame, text="Tensile Testing")

        # Create left panel for controls
        left_panel = ttk.Frame(tensile_frame, width=300)
        left_panel.pack(side='left', fill='y', padx=10, pady=10)
        left_panel.pack_propagate(False)

        # File loading section
        file_frame = ttk.LabelFrame(left_panel, text="File Loading", padding=10)
        file_frame.pack(fill='x', pady=5)

        ttk.Button(file_frame, text="Load Individual Files",
                   command=self.load_tensile_files, width=25).pack(pady=3)

        ttk.Button(file_frame, text="Load from Folder",
                   command=self.load_tensile_folder, width=25).pack(pady=3)

        ttk.Button(file_frame, text="Clear All Data",
                   command=self.clear_tensile_data, width=25).pack(pady=3)

        # Sample info section
        sample_frame = ttk.LabelFrame(left_panel, text="Sample Information", padding=10)
        sample_frame.pack(fill='x', pady=5)

        ttk.Label(sample_frame, text="Base Sample Name:").pack(anchor='w')
        self.tensile_sample_name = tk.StringVar(value="Sample")
        ttk.Entry(sample_frame, textvariable=self.tensile_sample_name, width=30).pack(pady=2, fill='x')

        # Analysis controls
        analysis_frame = ttk.LabelFrame(left_panel, text="Analysis Controls", padding=10)
        analysis_frame.pack(fill='x', pady=5)

        ttk.Button(analysis_frame, text="Analyze Data",
                   command=self.analyze_tensile_data, width=25).pack(pady=3)

        ttk.Button(analysis_frame, text="Generate Plot",
                   command=self.plot_tensile_data, width=25).pack(pady=3)

        ttk.Button(analysis_frame, text="Export Results",
                   command=self.export_tensile_results, width=25).pack(pady=3)

        # Status section
        status_frame = ttk.LabelFrame(left_panel, text="Status", padding=10)
        status_frame.pack(fill='both', expand=True, pady=5)

        self.tensile_status = scrolledtext.ScrolledText(status_frame, height=10, width=35, font=('Consolas', 9))
        self.tensile_status.pack(fill='both', expand=True)

        # Right panel for plots and results
        right_panel = ttk.Frame(tensile_frame)
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        # Plot area
        self.tensile_plot_frame = ttk.LabelFrame(right_panel, text="Visualization", padding=10)
        self.tensile_plot_frame.pack(fill='both', expand=True)

    def create_tga_tab(self):
        """Create TGA analysis tab"""
        tga_frame = ttk.Frame(self.notebook)
        self.notebook.add(tga_frame, text="TGA Analysis")

        # Create left panel for controls
        left_panel = ttk.Frame(tga_frame, width=300)
        left_panel.pack(side='left', fill='y', padx=10, pady=10)
        left_panel.pack_propagate(False)

        # File loading section
        file_frame = ttk.LabelFrame(left_panel, text="File Loading", padding=10)
        file_frame.pack(fill='x', pady=5)

        ttk.Button(file_frame, text="Load TGA Files",
                   command=self.load_tga_files, width=25).pack(pady=3)

        ttk.Button(file_frame, text="Load from Folder",
                   command=self.load_tga_folder, width=25).pack(pady=3)

        ttk.Button(file_frame, text="Clear All Data",
                   command=self.clear_tga_data, width=25).pack(pady=3)

        # Analysis controls
        analysis_frame = ttk.LabelFrame(left_panel, text="Analysis Controls", padding=10)
        analysis_frame.pack(fill='x', pady=5)

        ttk.Button(analysis_frame, text="Analyze TGA Data",
                   command=self.analyze_tga_data, width=25).pack(pady=3)

        ttk.Button(analysis_frame, text="Generate Plots",
                   command=self.plot_tga_data, width=25).pack(pady=3)

        ttk.Button(analysis_frame, text="Export Results",
                   command=self.export_tga_results, width=25).pack(pady=3)

        # Status section
        status_frame = ttk.LabelFrame(left_panel, text="Status", padding=10)
        status_frame.pack(fill='both', expand=True, pady=5)

        self.tga_status = scrolledtext.ScrolledText(status_frame, height=10, width=35, font=('Consolas', 9))
        self.tga_status.pack(fill='both', expand=True)

        # Right panel for plots
        right_panel = ttk.Frame(tga_frame)
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        # Plot area
        self.tga_plot_frame = ttk.LabelFrame(right_panel, text="Visualization", padding=10)
        self.tga_plot_frame.pack(fill='both', expand=True)

    def create_results_tab(self):
        """Create results display tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")

        # Create notebook for different result types
        results_notebook = ttk.Notebook(results_frame)
        results_notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tensile results
        tensile_results_frame = ttk.Frame(results_notebook)
        results_notebook.add(tensile_results_frame, text="Tensile Results")

        self.tensile_results_text = scrolledtext.ScrolledText(tensile_results_frame,
                                                              font=('Courier', 10))
        self.tensile_results_text.pack(fill='both', expand=True, padx=10, pady=10)

        # TGA results
        tga_results_frame = ttk.Frame(results_notebook)
        results_notebook.add(tga_results_frame, text="TGA Results")

        self.tga_results_text = scrolledtext.ScrolledText(tga_results_frame,
                                                          font=('Courier', 10))
        self.tga_results_text.pack(fill='both', expand=True, padx=10, pady=10)

    # Utility Methods
    def log_status(self, text_widget, message):
        """Log message to status widget"""
        text_widget.insert(tk.END, message + "\n")
        text_widget.see(tk.END)
        text_widget.update()

    # Tensile Testing Methods
    def clear_tensile_data(self):
        """Clear all tensile data"""
        self.tensile_data = {}
        self.tensile_status.delete(1.0, tk.END)
        self.tensile_results_text.delete(1.0, tk.END)
        for widget in self.tensile_plot_frame.winfo_children():
            widget.destroy()
        self.log_status(self.tensile_status, "All tensile data cleared")

    def load_tensile_files(self):
        """Load individual tensile testing files"""
        files = filedialog.askopenfilenames(
            title="Select Tensile Testing Files",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if files:
            self.process_tensile_files(files)

    def load_tensile_folder(self):
        """Load tensile files from folder"""
        folder = filedialog.askdirectory(title="Select Folder with Tensile Files")

        if folder:
            files = []
            for ext in ['*.txt', '*.csv']:
                files.extend(Path(folder).glob(ext))

            if files:
                self.process_tensile_files([str(f) for f in files])
            else:
                messagebox.showwarning("No Files", "No tensile data files found in selected folder")

    def process_tensile_files(self, files):
        """Process tensile testing files"""

        def process():
            self.tensile_status.delete(1.0, tk.END)
            self.log_status(self.tensile_status, f"Processing {len(files)} files...")

            success_count = 0
            base_name = self.tensile_sample_name.get()

            for i, filepath in enumerate(files):
                filename = Path(filepath).name
                trial_name = f"Run {i + 1}"

                self.log_status(self.tensile_status, f"\nProcessing: {filename}")

                # Get parameters from user
                params = self.get_tensile_parameters(filename)
                if params:
                    if self.load_single_tensile_file(filepath, base_name, trial_name, params):
                        success_count += 1
                        self.log_status(self.tensile_status, f"Successfully loaded {trial_name}")
                    else:
                        self.log_status(self.tensile_status, f"Failed to load {trial_name}")
                else:
                    self.log_status(self.tensile_status, f"Cancelled loading {trial_name}")

            self.log_status(self.tensile_status, f"\nCompleted: {success_count}/{len(files)} files loaded")

        # Run in separate thread to prevent GUI freezing
        threading.Thread(target=process, daemon=True).start()

    def get_tensile_parameters(self, filename):
        """Get tensile testing parameters from user"""
        dialog = ParameterDialog(self.root, f"Parameters for {filename}")
        self.root.wait_window(dialog.dialog)
        return dialog.result

    def load_single_tensile_file(self, filepath, sample_name, trial_name, params):
        """Load single tensile file"""
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Find data start
            data_start = 0
            for i, line in enumerate(lines):
                line_clean = line.strip().lower()
                if line_clean and (line_clean[0].isdigit() or line_clean[0] == '-'):
                    data_start = i
                    break

            # Parse data
            data_rows = []
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                try:
                    for delimiter in ['\t', ',', ' ']:
                        parts = line.replace(',', '.').split(delimiter)
                        parts = [p.strip() for p in parts if p.strip()]

                        if len(parts) >= 3:
                            crosshead = float(parts[0])
                            load = float(parts[1])
                            time = float(parts[2])
                            data_rows.append([crosshead, load, time])
                            break
                except (ValueError, IndexError):
                    continue

            if len(data_rows) < 10:
                return False

            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=['crosshead', 'load', 'time'])

            # Calculate stress and strain
            df['strain'] = df['crosshead'] / params['gauge_length']
            df['stress'] = df['load'] / params['cross_section_area']

            # Calculate properties
            properties = self.calculate_tensile_properties(df)

            # Store data
            full_name = f"{sample_name}_{trial_name}"
            self.tensile_data[full_name] = {
                'data': df,
                'properties': properties,
                'sample_name': sample_name,
                'trial_name': trial_name,
                'parameters': params
            }

            return True

        except Exception as e:
            self.log_status(self.tensile_status, f"Error: {e}")
            return False

    def calculate_tensile_properties(self, data, strain_range=None):
        """Calculate tensile properties with adaptive strain range"""
        properties = {}

        # Adaptive strain range for Young's modulus calculation
        max_strain = data['strain'].max()

        if strain_range is None:
            if max_strain > 5:  # High elongation material (>500%)
                strain_range = (0.01, 0.05)  # 1% to 5%
            elif max_strain > 1:  # Medium elongation (>100%)
                strain_range = (0.005, 0.02)  # 0.5% to 2%
            else:  # Low elongation material
                strain_range = (0.001, 0.005)  # 0.1% to 0.5%

        # Young's modulus calculation
        mask = (data['strain'] >= strain_range[0]) & (data['strain'] <= strain_range[1])
        linear_data = data[mask]

        if len(linear_data) >= 5:
            try:
                slope, _, r_value, _, _ = stats.linregress(linear_data['strain'], linear_data['stress'])
                properties['Youngs_Modulus_MPa'] = abs(slope)  # Take absolute value
                properties['R_squared'] = r_value ** 2
            except:
                properties['Youngs_Modulus_MPa'] = 0
                properties['R_squared'] = 0
        else:
            # Try with first 10% of data if linear range is too small
            early_data = data[data['strain'] <= max_strain * 0.1]
            if len(early_data) >= 5:
                try:
                    slope, _, r_value, _, _ = stats.linregress(early_data['strain'], early_data['stress'])
                    properties['Youngs_Modulus_MPa'] = abs(slope)
                    properties['R_squared'] = r_value ** 2
                except:
                    properties['Youngs_Modulus_MPa'] = 0
                    properties['R_squared'] = 0
            else:
                properties['Youngs_Modulus_MPa'] = 0
                properties['R_squared'] = 0

        # Other properties
        properties['UTS_MPa'] = data['stress'].max()
        properties['Strain_at_Break_percent'] = data['strain'].iloc[-1] * 100

        # Toughness calculation (fixed)
        try:
            properties['Toughness_MJ_per_m3'] = np.trapezoid(data['stress'], data['strain'])
        except AttributeError:
            properties['Toughness_MJ_per_m3'] = np.trapz(data['stress'], data['strain'])
        except:
            properties['Toughness_MJ_per_m3'] = 0

        return properties

    def analyze_tensile_data(self):
        """Analyze tensile testing data and generate publication table"""
        if not self.tensile_data:
            messagebox.showwarning("No Data", "Please load tensile data first")
            return

        # Group by sample
        sample_groups = {}
        for full_name, data_info in self.tensile_data.items():
            sample_name = data_info['sample_name']
            if sample_name not in sample_groups:
                sample_groups[sample_name] = []
            sample_groups[sample_name].append(data_info['properties'])

        # Calculate statistics
        results = []
        for sample_name, trials in sample_groups.items():
            properties = ['Youngs_Modulus_MPa', 'UTS_MPa', 'Strain_at_Break_percent', 'Toughness_MJ_per_m3']

            sample_stats = {'Sample': sample_name, 'n_trials': len(trials)}

            for prop in properties:
                values = [trial[prop] for trial in trials]
                sample_stats[f'{prop}_mean'] = np.mean(values)
                sample_stats[f'{prop}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                sample_stats[f'{prop}_cv'] = (np.std(values, ddof=1) / np.mean(values) * 100) if len(
                    values) > 1 and np.mean(values) != 0 else 0.0

            results.append(sample_stats)

        # Store results for export
        self.tensile_analysis_results = results

        # Display results
        self.display_tensile_results(results)
        self.log_status(self.tensile_status, "Analysis completed successfully")

    def display_tensile_results(self, results):
        """Display tensile analysis results in publication format"""
        self.tensile_results_text.delete(1.0, tk.END)

        output = "TENSILE TESTING ANALYSIS RESULTS\n"
        output += "=" * 90 + "\n\n"

        output += "Summary of engineering stress data.\n"
        output += "-" * 90 + "\n"
        output += f"{'Polyol':<20} {'Break strength':<16} {'Young\'s':<14} {'Toughness':<14} {'% Strain':<12} {'n':<4}\n"
        output += f"{'':20} {'(MPa)':<16} {'modulus':<14} {'(MJ/m³)':<14} {'':12} {'':4}\n"
        output += f"{'':20} {'':16} {'(MPa)':<14} {'':14} {'':12} {'':4}\n"
        output += "-" * 90 + "\n"

        for result in results:
            sample = result['Sample'][:19]
            n = int(result['n_trials'])

            if n > 1:
                # Format with mean ± std
                break_str = f"{result['UTS_MPa_mean']:.2f} ± {result['UTS_MPa_std']:.2f}"
                youngs_str = f"{result['Youngs_Modulus_MPa_mean']:.2f} ± {result['Youngs_Modulus_MPa_std']:.2f}"
                tough_str = f"{result['Toughness_MJ_per_m3_mean']:.2f} ± {result['Toughness_MJ_per_m3_std']:.2f}"
                strain_str = f"{result['Strain_at_Break_percent_mean']:.0f} ± {result['Strain_at_Break_percent_std']:.0f}"
            else:
                # Single measurement
                break_str = f"{result['UTS_MPa_mean']:.2f}"
                youngs_str = f"{result['Youngs_Modulus_MPa_mean']:.2f}"
                tough_str = f"{result['Toughness_MJ_per_m3_mean']:.2f}"
                strain_str = f"{result['Strain_at_Break_percent_mean']:.0f}"

            output += f"{sample:<20} {break_str:<16} {youngs_str:<14} {tough_str:<14} {strain_str:<12} {n:<4}\n"

        output += "-" * 90 + "\n"
        if any(r['n_trials'] > 1 for r in results):
            output += "Values are presented as mean ± standard deviation where n > 1\n"
        output += "=" * 90 + "\n"

        self.tensile_results_text.insert(1.0, output)


    def plot_tensile_data(self):
        """Plot tensile testing data"""
        if not self.tensile_data:
            messagebox.showwarning("No Data", "Please load tensile data first")
            return

        # Clear previous plot
        for widget in self.tensile_plot_frame.winfo_children():
            widget.destroy()

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
        line_styles = ['-', '--', '-.', ':', '-', '--']

        for i, (name, data_info) in enumerate(self.tensile_data.items()):
            data = data_info['data']
            trial_name = data_info['trial_name']

            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]

            ax.plot(data['strain'] * 100, data['stress'],
                    color=color, linestyle=line_style, linewidth=2.5,
                    label=trial_name, alpha=0.9)

        ax.set_xlabel('Strain (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stress (MPa)', fontsize=12, fontweight='bold')
        ax.set_title('Tensile Testing Results', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set clean borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        ax.tick_params(labelsize=11, width=1.2)

        plt.tight_layout()

        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, self.tensile_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        self.log_status(self.tensile_status, "Plot generated successfully")

    def export_tensile_results(self):
        """Export tensile results in publication format"""
        if not self.tensile_data:
            messagebox.showwarning("No Data", "Please analyze data first")
            return

        # Make sure analysis is done
        if not hasattr(self, 'tensile_analysis_results'):
            self.analyze_tensile_data()

        filename = filedialog.asksaveasfilename(
            title="Save Tensile Results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if filename:
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:

                    # 1. Publication Summary Table
                    pub_summary = []
                    for result in self.tensile_analysis_results:
                        n = int(result['n_trials'])

                        if n > 1:
                            pub_summary.append({
                                'Polyol': result['Sample'],
                                'Break strength (MPa)': f"{result['UTS_MPa_mean']:.2f} ± {result['UTS_MPa_std']:.2f}",
                                'Young\'s modulus (MPa)': f"{result['Youngs_Modulus_MPa_mean']:.2f} ± {result['Youngs_Modulus_MPa_std']:.2f}",
                                'Toughness (MJ/m³)': f"{result['Toughness_MJ_per_m3_mean']:.2f} ± {result['Toughness_MJ_per_m3_std']:.2f}",
                                '% Strain': f"{result['Strain_at_Break_percent_mean']:.0f} ± {result['Strain_at_Break_percent_std']:.0f}",
                                'n': n
                            })
                        else:
                            pub_summary.append({
                                'Polyol': result['Sample'],
                                'Break strength (MPa)': f"{result['UTS_MPa_mean']:.2f}",
                                'Young\'s modulus (MPa)': f"{result['Youngs_Modulus_MPa_mean']:.2f}",
                                'Toughness (MJ/m³)': f"{result['Toughness_MJ_per_m3_mean']:.2f}",
                                '% Strain': f"{result['Strain_at_Break_percent_mean']:.0f}",
                                'n': n
                            })

                    pd.DataFrame(pub_summary).to_excel(writer, sheet_name='Publication_Summary', index=False)

                    # 2. Statistical Summary (numerical values for further analysis)
                    stat_summary = []
                    for result in self.tensile_analysis_results:
                        stat_summary.append({
                            'Sample': result['Sample'],
                            'n_trials': result['n_trials'],
                            'UTS_mean_MPa': result['UTS_MPa_mean'],
                            'UTS_std_MPa': result['UTS_MPa_std'],
                            'UTS_CV_percent': result['UTS_MPa_cv'],
                            'Youngs_Modulus_mean_MPa': result['Youngs_Modulus_MPa_mean'],
                            'Youngs_Modulus_std_MPa': result['Youngs_Modulus_MPa_std'],
                            'Youngs_Modulus_CV_percent': result['Youngs_Modulus_MPa_cv'],
                            'Toughness_mean_MJ_per_m3': result['Toughness_MJ_per_m3_mean'],
                            'Toughness_std_MJ_per_m3': result['Toughness_MJ_per_m3_std'],
                            'Toughness_CV_percent': result['Toughness_MJ_per_m3_cv'],
                            'Strain_at_Break_mean_percent': result['Strain_at_Break_percent_mean'],
                            'Strain_at_Break_std_percent': result['Strain_at_Break_percent_std'],
                            'Strain_at_Break_CV_percent': result['Strain_at_Break_percent_cv']
                        })

                    pd.DataFrame(stat_summary).to_excel(writer, sheet_name='Statistical_Summary', index=False)

                    # 3. Individual Trials (raw data)
                    individual_data = []
                    for name, data_info in self.tensile_data.items():
                        props = data_info['properties'].copy()
                        props['Full_Name'] = name
                        props['Sample'] = data_info['sample_name']
                        props['Trial'] = data_info['trial_name']
                        props['Gauge_Length_mm'] = data_info['parameters']['gauge_length']
                        props['Cross_Section_Area_mm2'] = data_info['parameters']['cross_section_area']
                        individual_data.append(props)

                    pd.DataFrame(individual_data).to_excel(writer, sheet_name='Individual_Trials', index=False)

                messagebox.showinfo("Success", f"Results exported to {filename}")
                self.log_status(self.tensile_status, f"Publication-ready results exported to {Path(filename).name}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")

    # TGA Methods
    def clear_tga_data(self):
        """Clear all TGA data"""
        self.tga_data = {}
        self.tga_status.delete(1.0, tk.END)
        self.tga_results_text.delete(1.0, tk.END)
        for widget in self.tga_plot_frame.winfo_children():
            widget.destroy()
        self.log_status(self.tga_status, "All TGA data cleared")

    def load_tga_files(self):
        """Load TGA files"""
        files = filedialog.askopenfilenames(
            title="Select TGA Files",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if files:
            self.process_tga_files(files)

    def load_tga_folder(self):
        """Load TGA files from folder"""
        folder = filedialog.askdirectory(title="Select Folder with TGA Files")

        if folder:
            files = list(Path(folder).glob("*.csv"))

            if files:
                self.process_tga_files([str(f) for f in files])
            else:
                messagebox.showwarning("No Files", "No TGA files found in selected folder")

    def process_tga_files(self, files):
        """Process TGA files"""

        def process():
            self.tga_status.delete(1.0, tk.END)
            self.log_status(self.tga_status, f"Processing {len(files)} TGA files...")

            success_count = 0

            for filepath in files:
                filename = Path(filepath).name
                sample_name = self.extract_tga_sample_name(filename)

                self.log_status(self.tga_status, f"\nProcessing: {filename}")
                self.log_status(self.tga_status, f"Sample: {sample_name}")

                if self.load_single_tga_file(filepath, sample_name):
                    success_count += 1
                    self.log_status(self.tga_status, f"Successfully loaded")
                else:
                    self.log_status(self.tga_status, f"Failed to load")

            self.log_status(self.tga_status, f"\nCompleted: {success_count}/{len(files)} files loaded")

        threading.Thread(target=process, daemon=True).start()

    def extract_tga_sample_name(self, filename):
        """Extract sample name from TGA filename"""
        base_name = Path(filename).stem
        parts = base_name.split('_')

        sample_parts = []
        for part in parts:
            if 'mek' in part.lower():
                sample_parts.append('MEK')
            elif '%' in part:
                sample_parts.append(part)
            elif 'fabric' in part.lower():
                sample_parts.append('Fabric')
            elif 'bulk' in part.lower():
                sample_parts.append('Bulk')

        return '-'.join(sample_parts) if sample_parts else base_name

    def load_single_tga_file(self, filepath, sample_name):
        """Load single TGA file"""
        try:
            df = pd.read_csv(filepath)

            # Extract data
            time = df['Time'].values
            weight = df['Unsubtracted Weight'].values
            temperature = df['Sample Temperature'].values

            # Clean data
            valid_mask = ~(np.isnan(time) | np.isnan(weight) | np.isnan(temperature))
            time = time[valid_mask]
            weight = weight[valid_mask]
            temperature = temperature[valid_mask]

            # Convert to percentage
            weight_percent = (weight / weight[0]) * 100

            # Calculate derivative
            deriv_weight = np.gradient(weight_percent, temperature)

            # Analyze thermal events
            results = self.analyze_tga_thermal_events(temperature, weight_percent, deriv_weight)

            # Store data
            self.tga_data[sample_name] = {
                'temperature': temperature,
                'weight_percent': weight_percent,
                'deriv_weight': deriv_weight,
                'results': results,
                'filepath': filepath
            }

            self.log_status(self.tga_status,
                            f"  T5: {results['T5']:.0f}°C, T50: {results['T50']:.0f}°C, Tmax: {results['Tmax']:.0f}°C")

            return True

        except Exception as e:
            self.log_status(self.tga_status, f"  Error: {e}")
            return False

    def analyze_tga_thermal_events(self, temperature, weight_percent, deriv_weight):
        """Analyze TGA thermal events"""
        results = {}

        # T5
        try:
            indices_95 = np.where(weight_percent <= 95.0)[0]
            results['T5'] = temperature[indices_95[0]] if len(indices_95) > 0 else np.nan
        except:
            results['T5'] = np.nan

        # T50
        try:
            indices_50 = np.where(weight_percent <= 50.0)[0]
            results['T50'] = temperature[indices_50[0]] if len(indices_50) > 0 else np.nan
        except:
            results['T50'] = np.nan

        # Tmax
        try:
            decomp_mask = (temperature >= 200) & (temperature <= 600)
            if np.any(decomp_mask):
                decomp_deriv = deriv_weight[decomp_mask]
                decomp_temp = temperature[decomp_mask]
                min_deriv_idx = np.argmin(decomp_deriv)
                results['Tmax'] = decomp_temp[min_deriv_idx]
            else:
                results['Tmax'] = np.nan
        except:
            results['Tmax'] = np.nan

        # Residue
        try:
            if temperature.max() >= 600:
                idx_600 = np.where(temperature >= 600)[0]
                results['Residue_600C'] = weight_percent[idx_600[0]] if len(idx_600) > 0 else weight_percent[-1]
            else:
                results['Residue_600C'] = weight_percent[-1]
        except:
            results['Residue_600C'] = weight_percent[-1] if len(weight_percent) > 0 else np.nan

        return results

    def analyze_tga_data(self):
        """Analyze TGA data and display results"""
        if not self.tga_data:
            messagebox.showwarning("No Data", "Please load TGA data first")
            return

        self.display_tga_results()
        self.log_status(self.tga_status, "TGA analysis completed successfully")

    def display_tga_results(self):
        """Display TGA analysis results"""
        self.tga_results_text.delete(1.0, tk.END)

        output = "TGA ANALYSIS RESULTS\n"
        output += "=" * 80 + "\n\n"

        output += "TGA Data Summary\n"
        output += "-" * 80 + "\n"
        output += f"{'Sample':<25} {'T5 [°C]':<8} {'T50 [°C]':<9} {'Tmax [°C]':<10} {'Residue [%]':<12}\n"
        output += "-" * 80 + "\n"

        for sample_name, data in self.tga_data.items():
            results = data['results']
            sample = sample_name[:24]
            t5 = f"{results['T5']:.0f}" if not np.isnan(results['T5']) else "N/A"
            t50 = f"{results['T50']:.0f}" if not np.isnan(results['T50']) else "N/A"
            tmax = f"{results['Tmax']:.0f}" if not np.isnan(results['Tmax']) else "N/A"
            residue = f"{results['Residue_600C']:.1f}" if not np.isnan(results['Residue_600C']) else "N/A"

            output += f"{sample:<25} {t5:<8} {t50:<9} {tmax:<10} {residue:<12}\n"

        output += "-" * 80 + "\n"

        self.tga_results_text.insert(1.0, output)

    def plot_tga_data(self):
        """Plot TGA data"""
        if not self.tga_data:
            messagebox.showwarning("No Data", "Please load TGA data first")
            return

        # Clear previous plot
        for widget in self.tga_plot_frame.winfo_children():
            widget.destroy()

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#17becf', '#bcbd22']
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

        # Plot weight loss and derivative curves
        for i, (sample_name, data) in enumerate(self.tga_data.items()):
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]

            # Weight loss plot
            ax1.plot(data['temperature'], data['weight_percent'],
                     color=color, linestyle=line_style, linewidth=2.5,
                     label=sample_name, alpha=0.9)

            # Derivative plot
            ax2.plot(data['temperature'], -data['deriv_weight'],
                     color=color, linestyle=line_style, linewidth=2.5,
                     label=sample_name, alpha=0.9)

        # Format weight loss plot
        ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Weight %', fontsize=12, fontweight='bold')
        ax1.set_title('TGA Weight Loss', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 600)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Format derivative plot
        ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Deriv. Weight (%/°C)', fontsize=12, fontweight='bold')
        ax2.set_title('TGA Derivative', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 700)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        # Set clean borders
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')
            ax.tick_params(labelsize=11, width=1.2)

        plt.tight_layout()

        # Embed plot in GUI
        canvas = FigureCanvasTkAgg(fig, self.tga_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        self.log_status(self.tga_status, "TGA plots generated successfully")

    def export_tga_results(self):
        """Export TGA results to Excel"""
        if not self.tga_data:
            messagebox.showwarning("No Data", "Please analyze TGA data first")
            return

        filename = filedialog.asksaveasfilename(
            title="Save TGA Results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if filename:
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Summary table
                    summary_data = []
                    for sample_name, data in self.tga_data.items():
                        results = data['results']
                        summary_data.append({
                            'Sample': sample_name,
                            'T5_C': results['T5'],
                            'T50_C': results['T50'],
                            'Tmax_C': results['Tmax'],
                            'Residue_600C_percent': results['Residue_600C']
                        })

                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='TGA_Summary', index=False)

                    # Raw data for each sample
                    for sample_name, data in self.tga_data.items():
                        df_raw = pd.DataFrame({
                            'Temperature_C': data['temperature'],
                            'Weight_percent': data['weight_percent'],
                            'Deriv_Weight': data['deriv_weight']
                        })

                        sheet_name = sample_name.replace(' ', '_').replace('%', 'pct')[:31]
                        df_raw.to_excel(writer, sheet_name=sheet_name, index=False)

                messagebox.showinfo("Success", f"TGA results exported to {filename}")
                self.log_status(self.tga_status, f"Results exported to {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = MaterialsAnalyzerApp(root)

    # Set application icon (optional)
    try:
        root.iconbitmap('icon.ico')  # Add your icon file if available
    except:
        pass

    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()