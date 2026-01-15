"""
颜值评分系统 - GUI界面
Tkinter-based GUI for beauty scoring system
支持几何特征评分和深度学习评分
Supports geometric features and deep learning scoring
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

# Add project root to path (gui.py is now in gui_tool/ subdirectory)
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from analyzer import FaceAnalyzer
from scorer import FaceScorer


class TranslationManager:
    """Manages bilingual translations"""

    TRANSLATIONS = {
        'zh': {
            # Window
            'window_title': '颜值评分系统',
            # Control Panel
            'language': '语言',
            'scoring_mode': '评分模式',
            'geometric_mode': '几何特征',
            'dl_mode': '深度学习',
            'both_mode': '两者都',
            # Image Selection
            'select_image': '选择图片',
            'image_path': '图片路径',
            'filename': '文件名',
            'preview': '预览',
            # Model Selection
            'select_model': '选择模型',
            'model_path': '模型路径',
            'default_model': '默认模型 (best)',
            'device': '设备',
            # Actions
            'score_button': '开始评分',
            'clear_button': '清空',
            # Results
            'results': '评分结果',
            'final_score': '最终得分',
            'detailed_scores': '详细得分',
            'three_regions': '三庭比例',
            'five_eyes': '五眼比例',
            'symmetry': '对称性',
            'model_info': '模型信息',
            # Status
            'analyzing': '正在分析...',
            'loading_model': '正在加载模型...',
            'scoring': '正在评分...',
            'ready': '就绪',
            'success': '评分完成',
            'geometric_analysis': '几何特征分析中...',
            'dl_prediction': '深度学习预测中...',
            # Errors
            'error': '错误',
            'no_image': '请先选择图片',
            'no_model': '请先选择模型',
            'invalid_image': '无法读取图片',
            'no_face': '未检测到人脸',
            'model_error': '模型加载失败',
            'torch_not_available': 'PyTorch未安装，无法使用深度学习模式',
            # File types
            'image_files': '图片文件',
            'model_files': '模型文件',
            # Report headers
            'report_header': '=== 颜值评分报告（几何特征） ===',
            'three_regions_score': '三庭评分',
            'five_eyes_score': '五眼评分',
            'symmetry_score': '对称性评分',
            'overall_score': '综合评分',
            'weights': '权重',
            'analysis_details': '【分析详情】'
        },
        'en': {
            # Window
            'window_title': 'Beauty Score System',
            # Control Panel
            'language': 'Language',
            'scoring_mode': 'Scoring Mode',
            'geometric_mode': 'Geometric',
            'dl_mode': 'Deep Learning',
            'both_mode': 'Both',
            # Image Selection
            'select_image': 'Select Image',
            'image_path': 'Image Path',
            'filename': 'Filename',
            'preview': 'Preview',
            # Model Selection
            'select_model': 'Select Model',
            'model_path': 'Model Path',
            'default_model': 'Default Model (best)',
            'device': 'Device',
            # Actions
            'score_button': 'Start Scoring',
            'clear_button': 'Clear',
            # Results
            'results': 'Scoring Results',
            'final_score': 'Final Score',
            'detailed_scores': 'Detailed Scores',
            'three_regions': 'Three Regions',
            'five_eyes': 'Five Eyes',
            'symmetry': 'Symmetry',
            'model_info': 'Model Info',
            # Status
            'analyzing': 'Analyzing...',
            'loading_model': 'Loading model...',
            'scoring': 'Scoring...',
            'ready': 'Ready',
            'success': 'Scoring Complete',
            'geometric_analysis': 'Geometric analysis...',
            'dl_prediction': 'Deep learning prediction...',
            # Errors
            'error': 'Error',
            'no_image': 'Please select an image first',
            'no_model': 'Please select a model first',
            'invalid_image': 'Cannot read image',
            'no_face': 'No face detected',
            'model_error': 'Model loading failed',
            'torch_not_available': 'PyTorch not installed, cannot use deep learning mode',
            # File types
            'image_files': 'Image Files',
            'model_files': 'Model Files',
            # Report headers
            'report_header': '=== Beauty Score Report (Geometric) ===',
            'three_regions_score': 'Three Regions Score',
            'five_eyes_score': 'Five Eyes Score',
            'symmetry_score': 'Symmetry Score',
            'overall_score': 'Overall Score',
            'weights': 'Weights',
            'analysis_details': '[Analysis Details]'
        }
    }

    def __init__(self, default_lang='zh'):
        self.current_lang = default_lang

    def get(self, key):
        """Get translation for key"""
        return self.TRANSLATIONS[self.current_lang].get(key, key)

    def set_language(self, lang):
        """Set current language"""
        if lang in self.TRANSLATIONS:
            self.current_lang = lang


class BeautyScoreGUI:
    """Main GUI application class"""

    def __init__(self, root):
        self.root = root
        self.root.title("颜值评分系统 Beauty Score System")

        # Initialize components
        self.translator = TranslationManager('zh')
        self.scoring_mode = tk.StringVar(value='geometric')
        self.image_path = tk.StringVar()
        self.model_path = tk.StringVar(value='outputs/checkpoints/resnet18_best.pth')
        self.current_language = 'zh'

        # Model cache (for deep learning mode)
        self.dl_model = None
        self.dl_model_name = None
        self.device = self._get_device()

        # Store widget references for language updates
        self.widgets = {}

        # Initialize analyzers (lazy loading)
        self.face_analyzer = None
        self.face_scorer = None

        # Image preview storage
        self.current_image_tk = None

        # Build GUI
        self.create_widgets()
        self.update_ui_language()

    def _get_device(self):
        """Get available device for deep learning"""
        if TORCH_AVAILABLE:
            try:
                import torch
                return 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                return 'cpu'
        return 'cpu'

    def create_widgets(self):
        """Create all GUI widgets"""

        # === Top Control Panel ===
        top_frame = ttk.LabelFrame(self.root, text="控制面板 Control Panel")
        top_frame.pack(fill='x', padx=10, pady=5)

        # Language button
        self.widgets['lang_btn'] = ttk.Button(
            top_frame,
            text="English",
            command=self.toggle_language
        )
        self.widgets['lang_btn'].pack(side='left', padx=5, pady=5)

        # Scoring mode radio buttons
        mode_frame = ttk.Frame(top_frame)
        mode_frame.pack(side='left', padx=20)

        ttk.Label(mode_frame, text="评分模式 Mode:").pack(side='left')

        self.widgets['mode_geo'] = ttk.Radiobutton(
            mode_frame,
            text="几何特征 Geometric",
            variable=self.scoring_mode,
            value='geometric',
            command=self.on_mode_change
        )
        self.widgets['mode_geo'].pack(side='left', padx=5)

        self.widgets['mode_dl'] = ttk.Radiobutton(
            mode_frame,
            text="深度学习 DL",
            variable=self.scoring_mode,
            value='dl',
            command=self.on_mode_change
        )
        self.widgets['mode_dl'].pack(side='left', padx=5)

        self.widgets['mode_both'] = ttk.Radiobutton(
            mode_frame,
            text="两者都 Both",
            variable=self.scoring_mode,
            value='both',
            command=self.on_mode_change
        )
        self.widgets['mode_both'].pack(side='left', padx=5)

        # === Middle Input Section ===
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Left: Image selection
        image_frame = ttk.LabelFrame(middle_frame, text="图片 Image")
        image_frame.pack(side='left', fill='both', expand=True, padx=5)

        # Select image button
        self.widgets['select_img_btn'] = ttk.Button(
            image_frame,
            text="选择图片 Select Image",
            command=self.select_image
        )
        self.widgets['select_img_btn'].pack(pady=5)

        # Image path entry
        ttk.Label(image_frame, text="路径 Path:").pack(anchor='w', padx=5)
        self.image_entry = ttk.Entry(image_frame, textvariable=self.image_path)
        self.image_entry.pack(fill='x', padx=5, pady=2)

        # Filename display
        ttk.Label(image_frame, text="文件名 Filename:").pack(anchor='w', padx=5)
        self.filename_label = ttk.Label(image_frame, text="-", relief='sunken', anchor='w')
        self.filename_label.pack(fill='x', padx=5, pady=2)
        self.widgets['filename'] = self.filename_label

        # Image preview
        ttk.Label(image_frame, text="预览 Preview:").pack(anchor='w', padx=5)

        # Create a container frame for the preview with border
        preview_container = tk.Frame(image_frame, relief='solid', borderwidth=1, bg='white')
        preview_container.pack(padx=5, pady=5, fill='both', expand=True)

        # Preview label inside container - no fixed size constraints
        self.preview_label = tk.Label(
            preview_container,
            text="No Image",
            bg='white',
            anchor='center'
        )
        # Center the label in the container
        self.preview_label.place(relx=0.5, rely=0.5, anchor='center')
        self.widgets['preview'] = self.preview_label

        # Right: Model selection (for DL mode)
        model_frame = ttk.LabelFrame(middle_frame, text="模型 Model")
        model_frame.pack(side='right', fill='both', expand=True, padx=5)

        # Select model button
        self.widgets['select_model_btn'] = ttk.Button(
            model_frame,
            text="选择模型 Select Model",
            command=self.select_model
        )
        self.widgets['select_model_btn'].pack(pady=5)

        # Model path entry
        ttk.Label(model_frame, text="路径 Path:").pack(anchor='w', padx=5)
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path)
        self.model_entry.pack(fill='x', padx=5, pady=2)

        # Device info
        self.device_label = ttk.Label(
            model_frame,
            text=f"设备 Device: {self.device}"
        )
        self.device_label.pack(anchor='w', padx=5, pady=5)
        self.widgets['device'] = self.device_label

        # Default model hint
        hint_label = ttk.Label(
            model_frame,
            text="默认: outputs/checkpoints/\nresnet18_best.pth",
            foreground='gray',
            font=('Arial', 8)
        )
        hint_label.pack(anchor='w', padx=5, pady=5)

        # === Bottom Results Section ===
        results_frame = ttk.LabelFrame(self.root, text="结果 Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Score display
        score_display_frame = ttk.Frame(results_frame)
        score_display_frame.pack(fill='x', pady=5)

        ttk.Label(score_display_frame, text="最终得分 Final Score:").pack(side='left', padx=5)
        self.score_label = ttk.Label(
            score_display_frame,
            text="-",
            font=('Arial', 14, 'bold'),
            foreground='black'
        )
        self.score_label.pack(side='left', padx=10)
        self.widgets['final_score'] = self.score_label

        # Detailed scores text area
        ttk.Label(results_frame, text="详细得分 Detailed Scores:").pack(anchor='w', padx=5)
        self.details_text = tk.Text(results_frame, height=12, width=80, wrap='word')
        self.details_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.widgets['details'] = self.details_text

        # === Bottom Action Buttons ===
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill='x', padx=10, pady=5)

        self.widgets['score_btn'] = ttk.Button(
            action_frame,
            text="开始评分 Start Scoring",
            command=self.start_scoring
        )
        self.widgets['score_btn'].pack(side='left', padx=5)

        self.widgets['clear_btn'] = ttk.Button(
            action_frame,
            text="清空 Clear",
            command=self.clear_all
        )
        self.widgets['clear_btn'].pack(side='left', padx=5)

        # === Status Bar ===
        self.status_label = ttk.Label(
            self.root,
            text="就绪 Ready",
            relief='sunken',
            anchor='w'
        )
        self.status_label.pack(fill='x', padx=10, pady=(0, 5))
        self.widgets['status'] = self.status_label

    def toggle_language(self):
        """Toggle between Chinese and English"""
        self.current_language = 'en' if self.current_language == 'zh' else 'zh'
        self.translator.set_language(self.current_language)
        self.update_ui_language()

    def update_ui_language(self):
        """Update all UI text based on current language"""
        t = self.translator.get

        # Update window title
        title_zh = "颜值评分系统"
        title_en = "Beauty Score System"
        if self.current_language == 'zh':
            self.root.title(title_zh)
            self.widgets['lang_btn'].config(text="English")
        else:
            self.root.title(f"{title_en} - {title_zh}")
            self.widgets['lang_btn'].config(text="中文")

        # Update radio buttons
        self.widgets['mode_geo'].config(text=t('geometric_mode'))
        self.widgets['mode_dl'].config(text=t('dl_mode'))
        self.widgets['mode_both'].config(text=t('both_mode'))

        # Update buttons
        self.widgets['select_img_btn'].config(text=t('select_image'))
        self.widgets['select_model_btn'].config(text=t('select_model'))
        self.widgets['score_btn'].config(text=t('score_button'))
        self.widgets['clear_btn'].config(text=t('clear_button'))

        # Update device label
        self.widgets['device'].config(text=f"{t('device')}: {self.device}")

        # Update status
        current_status = self.status_label.cget('text')
        if '就绪' in current_status or 'Ready' in current_status:
            self.status_label.config(text=t('ready'))

    def on_mode_change(self):
        """Handle scoring mode change"""
        mode = self.scoring_mode.get()
        t = self.translator.get

        if mode == 'geometric':
            self.set_status(t('ready'))
        elif mode == 'dl':
            if not TORCH_AVAILABLE:
                messagebox.showwarning(
                    t('error'),
                    t('torch_not_available')
                )
                self.scoring_mode.set('geometric')
                self.set_status(t('ready'))
            else:
                self.set_status(t('ready'))
        else:  # both
            if not TORCH_AVAILABLE:
                messagebox.showwarning(
                    t('error'),
                    t('torch_not_available') + "\n" + "Switching to geometric mode only."
                )
                self.scoring_mode.set('geometric')
                self.set_status(t('ready'))
            else:
                self.set_status(t('ready'))

    def select_image(self):
        """Open file dialog to select image"""
        t = self.translator.get
        filetypes = [
            (t('image_files'), '*.jpg *.jpeg *.png *.bmp'),
            ('All Files', '*.*')
        ]

        filename = filedialog.askopenfilename(
            title=t('select_image'),
            filetypes=filetypes
        )

        if filename:
            self.image_path.set(filename)
            self.update_image_preview(filename)

    def update_image_preview(self, image_path):
        """Update image preview"""
        t = self.translator.get
        try:
            # Load image
            image = Image.open(image_path)

            # Get original dimensions
            orig_width, orig_height = image.size

            # Preview area max dimensions (in pixels)
            max_width = 350
            max_height = 450

            # Calculate scaling ratio to fit within preview area while maintaining aspect ratio
            width_ratio = max_width / orig_width
            height_ratio = max_height / orig_height
            scale_ratio = min(width_ratio, height_ratio)

            # Calculate new dimensions
            new_width = int(orig_width * scale_ratio)
            new_height = int(orig_height * scale_ratio)

            # Resize image with high quality
            image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_image_tk = ImageTk.PhotoImage(image_resized)

            # Update preview label - Clear text and show image
            self.preview_label.config(
                image=self.current_image_tk,
                text="",
                bg='white'
            )

            # Update filename display
            filename = os.path.basename(image_path)
            self.filename_label.config(text=filename)

            self.set_status(f"Image loaded: {filename} ({orig_width}x{orig_height}) -> ({new_width}x{new_height})")

        except Exception as e:
            messagebox.showerror(
                t('error'),
                f"{t('invalid_image')}: {str(e)}"
            )

    def select_model(self):
        """Open file dialog to select model"""
        t = self.translator.get
        filetypes = [
            (t('model_files'), '*.pth'),
            ('All Files', '*.*')
        ]

        # Check if default directory exists
        initial_dir = 'outputs/checkpoints'
        if not os.path.exists(initial_dir):
            initial_dir = None

        filename = filedialog.askopenfilename(
            title=t('select_model'),
            filetypes=filetypes,
            initialdir=initial_dir
        )

        if filename:
            self.model_path.set(filename)
            # Clear cached model when model changes
            self.dl_model = None
            self.dl_model_name = None
            self.set_status(f"Model selected: {os.path.basename(filename)}")

    def start_scoring(self):
        """Start scoring process in a separate thread"""
        t = self.translator.get

        # Validate inputs
        if not self.image_path.get():
            messagebox.showwarning(
                t('error'),
                t('no_image')
            )
            return

        if not os.path.exists(self.image_path.get()):
            messagebox.showwarning(
                t('error'),
                f"{t('invalid_image')}\n{self.image_path.get()}"
            )
            return

        mode = self.scoring_mode.get()
        if mode in ['dl', 'both']:
            if not TORCH_AVAILABLE:
                messagebox.showwarning(
                    t('error'),
                    t('torch_not_available')
                )
                return
            if not self.model_path.get():
                messagebox.showwarning(
                    t('error'),
                    t('no_model')
                )
                return
            if not os.path.exists(self.model_path.get()):
                messagebox.showwarning(
                    t('error'),
                    f"{t('model_error')}\n{self.model_path.get()}"
                )
                return

        # Disable buttons during scoring
        self.set_scoring_state(True)

        # Start scoring in background thread
        thread = threading.Thread(target=self._scoring_thread, args=(mode,))
        thread.daemon = True
        thread.start()

    def _scoring_thread(self, mode):
        """Scoring thread to avoid blocking UI"""
        t = self.translator.get
        try:
            results = {}

            # Geometric scoring
            if mode in ['geometric', 'both']:
                self.root.after(0, lambda: self.set_status(t('geometric_analysis')))
                results['geometric'] = self._geometric_scoring()

            # Deep learning scoring
            if mode in ['dl', 'both']:
                self.root.after(0, lambda: self.set_status(t('dl_prediction')))
                results['dl'] = self._dl_scoring()

            # Update UI with results (must be in main thread)
            self.root.after(0, lambda: self._display_results(results, mode))

        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
        finally:
            self.root.after(0, lambda: self.set_scoring_state(False))

    def _geometric_scoring(self):
        """Perform geometric feature scoring"""
        # Initialize analyzers if needed
        if self.face_analyzer is None:
            self.face_analyzer = FaceAnalyzer()
        if self.face_scorer is None:
            self.face_scorer = FaceScorer()

        # Analyze image
        analysis_result = self.face_analyzer.analyze(self.image_path.get())

        # Check if face was detected
        if not analysis_result.get('face_detected', False):
            raise ValueError(self.translator.get('no_face'))

        # Score
        scores = self.face_scorer.score(analysis_result)

        return scores

    def _dl_scoring(self):
        """Perform deep learning scoring"""
        from predict import load_model, predict

        # Predict
        score, score_100, model_name = predict(
            self.image_path.get(),
            self.model_path.get(),
            self.device
        )

        return {
            'score': score,
            'score_100': score_100,
            'model_name': model_name,
            'device': self.device
        }

    def _display_results(self, results, mode):
        """Display scoring results"""
        t = self.translator.get
        self.details_text.delete('1.0', 'end')

        final_score = 0
        report = ""

        if mode == 'geometric':
            geo = results['geometric']
            overall = geo['overall']
            final_score = overall['overall_score']

            # Parse and translate the report
            report = self._translate_geometric_report(geo['report'])

        elif mode == 'dl':
            dl = results['dl']
            final_score = dl['score']

            report = f"""=== {t('results')} ({t('dl_mode')}) ===

{t('model_info')}: {dl['model_name']}
{t('device')}: {dl['device']}

{t('final_score')}: {dl['score']:.2f}/5.0 ({dl['score_100']:.1f}/100)
"""

        elif mode == 'both':
            geo = results['geometric']
            dl = results['dl']

            geo_score = geo['overall']['overall_score']
            dl_score = dl['score']

            # Average of both methods
            final_score = (geo_score + dl_score) / 2

            report = f"""=== {t('results')} ({t('both_mode')}) ===

--- {t('geometric_mode')} ---
{geo_score:.2f}/5.0

--- {t('dl_mode')} ---
{dl['score']:.2f}/5.0 (Model: {dl['model_name']})

--- {t('final_score')} (Average) ---
{final_score:.2f}/5.0 ({final_score * 20:.1f}/100)

{'='*50}

{t('geometric_mode')} Details:
{self._translate_geometric_report(geo['report'])}
"""

        # Update final score display
        self.score_label.config(
            text=f"{final_score:.2f}/5.0\n({final_score * 20:.1f}/100)"
        )

        # Update details text
        self.details_text.insert('1.0', report)

        self.set_status(t('success'))

    def _translate_geometric_report(self, report):
        """Translate key terms in geometric report"""
        t = self.translator.get
        if self.current_language == 'zh':
            return report

        # Simple translation for key terms
        translations = {
            '=== 颜值评分报告（几何特征） ===': '=== Beauty Score Report (Geometric) ===',
            '【综合评分】': '[Overall Score]',
            '【三庭评分】': '[Three Regions Score]',
            '【五眼评分】': '[Five Eyes Score]',
            '【对称性评分】': '[Symmetry Score]',
            '【分析详情】': '[Analysis Details]',
            '得分:': 'Score:',
            '权重:': 'Weight:',
            '上庭': 'Upper region',
            '中庭': 'Middle region',
            '下庭': 'Lower region',
            '脸宽': 'Face width',
            '眼宽': 'Eye width',
            '比例': 'Ratio',
        }

        for zh, en in translations.items():
            report = report.replace(zh, en)

        return report

    def _show_error(self, error_msg):
        """Show error message"""
        t = self.translator.get
        messagebox.showerror(
            t('error'),
            error_msg
        )
        self.set_status(f"Error: {error_msg}")

    def set_scoring_state(self, is_scoring):
        """Enable/disable widgets during scoring"""
        state = 'disabled' if is_scoring else 'normal'
        self.widgets['score_btn'].config(state=state)
        self.widgets['select_img_btn'].config(state=state)
        self.widgets['select_model_btn'].config(state=state)
        self.widgets['clear_btn'].config(state=state)

    def set_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)

    def clear_all(self):
        """Clear all inputs and results"""
        t = self.translator.get
        self.image_path.set("")
        self.model_path.set("outputs/checkpoints/resnet18_best.pth")
        self.preview_label.config(image="", text="No Image")
        self.filename_label.config(text="-")
        self.score_label.config(text="-")
        self.details_text.delete('1.0', 'end')
        self.current_image_tk = None

        # Clear cached model
        self.dl_model = None
        self.dl_model_name = None

        self.set_status(t('ready'))


def main():
    """Main entry point"""
    root = tk.Tk()
    root.geometry("900x750")

    # Try to set window icon (optional)
    try:
        # root.iconbitmap('icon.ico')  # Uncomment if icon is available
        pass
    except Exception:
        pass

    app = BeautyScoreGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
