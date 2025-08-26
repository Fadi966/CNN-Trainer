import asyncio
import os
import threading
import re  # ← لاستخراج الأرقام من رسائل الحالة
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import tensorflow as tf
from PIL import Image
import io
from pathlib import Path

from cnn_model_handler import CNNModelHandler


class CNNTrainerApp(toga.App):
    def startup(self):
        self.model_handler = CNNModelHandler()

        self.current_model_path = None
        self.training_in_progress = False

        self.main_window = toga.MainWindow(title="CNN Trainer")

        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10))

        title_label = toga.Label(
            "CNN Training and Evaluation Tool",
            style=Pack(padding=(0, 0, 20, 0), font_size=20, text_align="center"),
        )

        description = toga.Label(
            "Train and evaluate CNN models with customizable parameters",
            style=Pack(padding=(0, 0, 20, 0), text_align="center"),
        )

        train_button = toga.Button(
            "Train New Model",
            on_press=self.open_training_window,
            style=Pack(padding=5, width=200),
        )

        evaluate_button = toga.Button(
            "Evaluate Model",
            on_press=self.open_evaluation_window,
            style=Pack(padding=5, width=200),
        )

        main_box.add(title_label)
        main_box.add(description)
        main_box.add(train_button)
        main_box.add(evaluate_button)

        self.main_window.content = main_box
        self.main_window.show()

        # حلقة الحدث باسم غير محجوز
        self._loop = asyncio.get_event_loop()

    def open_training_window(self, widget):
        self.training_window = toga.Window(title="Model Training")
        train_box = toga.Box(style=Pack(direction=COLUMN, margin=10))
        title_label = toga.Label("CNN Training Configuration", style=Pack(margin=(0, 0, 20, 0), font_size=16))
        train_box.add(title_label)

        data_label = toga.Label("Training Data:")
        self.data_input = toga.TextInput(readonly=True)
        data_button = toga.Button("Select Data Folder", on_press=self.select_training_data)
        data_box = toga.Box(style=Pack(direction=ROW, margin=5))
        data_box.add(data_label)
        data_box.add(self.data_input)
        data_box.add(data_button)
        train_box.add(data_box)

        params_label = toga.Label("Training Parameters:", style=Pack(margin=(10, 0, 5, 0), font_size=14))
        train_box.add(params_label)

        epochs_box = toga.Box(style=Pack(direction=ROW, margin=5))
        epochs_label = toga.Label("Epochs:")
        self.epochs_input = toga.NumberInput(style=Pack())
        self.epochs_input.value = 10
        epochs_box.add(epochs_label)
        epochs_box.add(self.epochs_input)
        train_box.add(epochs_box)

        batch_box = toga.Box(style=Pack(direction=ROW, margin=5))
        batch_label = toga.Label("Batch Size:")
        self.batch_input = toga.NumberInput(style=Pack())
        self.batch_input.value = 32
        batch_box.add(batch_label)
        batch_box.add(self.batch_input)
        train_box.add(batch_box)

        architecture_label = toga.Label("Model Architecture:", style=Pack(margin=(10, 0, 5, 0), font_size=14))
        train_box.add(architecture_label)

        self.model_selection = toga.Selection(items=["Custom CNN", "VGG16", "VGG19", "ResNet50", "GoogLeNet/Inception"])
        self.model_selection.on_change = self.on_model_selection_change
        train_box.add(self.model_selection)

        self.custom_box = toga.Box(style=Pack(direction=COLUMN, margin=5))

        conv_box = toga.Box(style=Pack(direction=ROW, margin=5))
        conv_label = toga.Label("Conv Layers:")
        self.conv_input = toga.NumberInput(style=Pack())
        self.conv_input.value = 3
        conv_box.add(conv_label)
        conv_box.add(self.conv_input)
        self.custom_box.add(conv_box)

        pool_box = toga.Box(style=Pack(direction=ROW, margin=5))
        pool_label = toga.Label("Pooling Layers:")
        self.pool_input = toga.NumberInput(style=Pack())
        self.pool_input.value = 3
        pool_box.add(pool_label)
        pool_box.add(self.pool_input)
        self.custom_box.add(pool_box)

        train_box.add(self.custom_box)

        # ===== لوحة معلومات التدريب الحيّة (إضافة) =====
        live_label = toga.Label("Live Training Info", style=Pack(margin=(10, 0, 5, 0), font_size=14))
        train_box.add(live_label)

        self.epoch_label = toga.Label("Epoch: - / -")
        self.metrics_label = toga.Label("acc: --  val_acc: --  |  loss: --  val_loss: --")
        self.progress = toga.ProgressBar(max=1, value=0, style=Pack(width=420))
        self.log_area = toga.MultilineTextInput(readonly=True, style=Pack(height=160))

        live_box = toga.Box(style=Pack(direction=COLUMN, margin=(0, 0, 10, 0)))
        live_box.add(self.epoch_label)
        live_box.add(self.metrics_label)
        live_box.add(self.progress)
        live_box.add(self.log_area)
        train_box.add(live_box)
        # ===============================================

        button_box = toga.Box(style=Pack(direction=ROW, margin=(20, 5)))
        self.train_model_button = toga.Button("Train Model", on_press=self.train_model, style=Pack(flex=1))
        self.save_model_button = toga.Button("Save Model", on_press=self.save_model, style=Pack(flex=1))
        self.save_model_button.enabled = False
        button_box.add(self.train_model_button)
        button_box.add(self.save_model_button)
        train_box.add(button_box)
        self.status_label = toga.Label("Ready", style=Pack(margin=(10, 5)))
        train_box.add(self.status_label)

        self.training_window.content = train_box
        self.training_window.show()
        self.custom_box.style.visibility = "hidden"
        self.on_model_selection_change(self.model_selection)

    def on_model_selection_change(self, widget):
        self.custom_box.style.visibility = "visible" if self.model_selection.value == "Custom CNN" else "hidden"

    def select_training_data(self, widget):
        async def run_dialog():
            try:
                dialog = toga.SelectFolderDialog("Select Training Data Folder")
                result = await self.main_window.dialog(dialog)
                if result:
                    self.data_input.value = result
            except Exception as e:
                err = toga.ErrorDialog("Error", f"Error selecting folder: {e}")
                await self.main_window.dialog(err)

        asyncio.ensure_future(run_dialog())

    # إضافة: دالة مساعدة لتدوين السجلّ
    def _append_log(self, msg: str):
        self.log_area.value = (self.log_area.value or "") + msg.rstrip() + "\n"

    # إضافة: تحديث مركزي للوحة من رسائل الحالة
    def _on_status_message(self, msg: str):
        self._update_status(msg)        # يحدّث شريط الحالة السفلي
        self._append_log(msg)           # يضيف للسجلّ
        self._parse_and_update_progress(msg)  # يحدّث الشريط/الأرقام إن وُجدت

    # إضافة: استخراج الأرقام من نصّ الرسالة (بدون تغيير في الـhandler)
    def _parse_and_update_progress(self, msg: str):
        # Epoch x / y
        m = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", msg)
        if m:
            e, E = int(m.group(1)), int(m.group(2))
            self.epoch_label.text = f"Epoch: {e} / {E}"
            self.progress.max = E
            self.progress.value = e

        # acc/val_acc/loss/val_loss إن كانت موجودة في النص
        # أمثلة محتملة من رسائلك: "acc=0.532"، "val_acc=0.750"
        m_acc = re.search(r"(?:acc(?:uracy)?=|acc:\s*)([0-9.]+)", msg)
        m_val = re.search(r"(?:val_acc=|val_accuracy[:=]\s*)([0-9.]+)", msg)
        m_loss = re.search(r"(?:loss[:=]\s*)([0-9.]+)", msg)
        m_vloss = re.search(r"(?:val_loss[:=]\s*)([0-9.]+)", msg)

        # ابقِ القيم السابقة إن لم توجد جديدة
        def _to_text(v, default):
            return f"{float(v):.4f}" if v is not None else default

        parts = self.metrics_label.text.split()
        acc_txt = _to_text(m_acc.group(1) if m_acc else None, parts[1] if len(parts) >= 2 else "--")
        val_txt = _to_text(m_val.group(1) if m_val else None, parts[3] if len(parts) >= 4 else "--")
        loss_txt = _to_text(m_loss.group(1) if m_loss else None, parts[6] if len(parts) >= 7 else "--")
        vloss_txt = _to_text(m_vloss.group(1) if m_vloss else None, parts[8] if len(parts) >= 9 else "--")

        self.metrics_label.text = f"acc: {acc_txt}  val_acc: {val_txt}  |  loss: {loss_txt}  val_loss: {vloss_txt}"

    def train_model(self, widget):
        """Train a CNN model with the specified parameters"""
        try:
            if not self.data_input.value:
                self.main_window.error_dialog(
                    "Input Error",
                    "Please select a training data folder"
                )
                return

            try:
                epochs = int(self.epochs_input.value)
                if epochs < 1 or epochs > 1000:
                    self.main_window.error_dialog(
                        "Input Error",
                        "Epochs must be between 1 and 1000"
                    )
                    return
            except (ValueError, TypeError):
                self.main_window.error_dialog(
                    "Input Error",
                    "Epochs must be a valid number"
                )
                return

            try:
                batch_size = int(self.batch_input.value)
                if batch_size < 1 or batch_size > 1024:
                    self.main_window.error_dialog(
                        "Input Error",
                        "Batch size must be between 1 and 1024"
                    )
                    return
            except (ValueError, TypeError):
                self.main_window.error_dialog(
                    "Input Error",
                    "Batch size must be a valid number"
                )
                return

            model_type = self.model_selection.value

            if model_type == "Custom CNN":
                try:
                    conv_layers = int(self.conv_input.value)
                    if conv_layers < 1 or conv_layers > 10:
                        self.main_window.error_dialog(
                            "Input Error",
                            "Number of convolutional layers must be between 1 and 10"
                        )
                        return
                except (ValueError, TypeError):
                    self.main_window.error_dialog(
                        "Input Error",
                        "Number of convolutional layers must be a valid number"
                    )
                    return

                try:
                    pool_layers = int(self.pool_input.value)
                    if pool_layers < 0 or pool_layers > 10:
                        self.main_window.error_dialog(
                            "Input Error",
                            "Number of pooling layers must be between 0 and 10"
                        )
                        return
                except (ValueError, TypeError):
                    self.main_window.error_dialog(
                        "Input Error",
                        "Number of pooling layers must be a valid number"
                    )
                    return

            # ↓↓↓ تهيئة اللوحة قبل بدء الخيط
            total_epochs = int(self.epochs_input.value)
            self.progress.max = total_epochs
            self.progress.value = 0
            self.epoch_label.text = f"Epoch: 0 / {total_epochs}"
            self.metrics_label.text = "acc: --  val_acc: --  |  loss: --  val_loss: --"
            self.log_area.value = ""
            # ↑↑↑

            self.train_model_button.enabled = False
            self.training_in_progress = True

            self.status_label.text = "Initializing model..."

            # Create the model in a background thread
            threading.Thread(
                target=self._train_model_thread,
                args=(model_type, epochs, batch_size),
                daemon=True
            ).start()

        except Exception as e:
            self.main_window.error_dialog(
                "Error",
                f"An error occurred while preparing training: {str(e)}"
            )
            self.train_model_button.enabled = True
            self.training_in_progress = False

    def cancel_training(self, widget):
        """Cancel the training process"""
        self.training_in_progress = False
        self.train_model_button.enabled = True

    def _update_status(self, message):
        print(message, flush=True)
        self.status_label.text = message

    def _train_model_thread(self, model_type, epochs, batch_size):
        print("TRAIN THREAD started", flush=True)

        """Background thread for model training"""
        try:
            self.invoke_later(lambda: self._update_status("Creating model..."))

            # Create the appropriate model
            if model_type == "Custom CNN":
                conv_layers = int(self.conv_input.value)
                pooling_layers = int(self.pool_input.value)

                num_classes = len([
                    name for name in os.listdir(self.data_input.value)
                    if os.path.isdir(os.path.join(self.data_input.value, name))
                ])

                self.model_handler.create_custom_model(
                    num_classes=num_classes,
                    conv_layers=conv_layers,
                    pooling_layers=pooling_layers
                )
            else:
                print("detecting classes")
                # Detect number of classes from the data directory
                num_classes = len([
                    name for name in os.listdir(self.data_input.value)
                    if os.path.isdir(os.path.join(self.data_input.value, name))
                ])

                print("creating model")
                self.model_handler.create_pretrained_model(
                    model_name=model_type,
                    num_classes=num_classes
                )

            self.invoke_later(lambda: self._update_status(f"Training model for {epochs} epochs..."))

            #Train the model — نرسل الرسائل للوحة عبر _on_status_message
            self.model_handler.train_model(
                data_path=self.data_input.value,
                epochs=epochs,
                batch_size=batch_size,
                status_callback=lambda msg: self.invoke_later(
                    lambda m=msg: self._on_status_message(m)
                ),
            )

            self.invoke_later(self._training_complete)

        except Exception as exc:
            msg = str(exc)
            self.invoke_later(lambda m=msg: self._training_error(m))

    # تصحيح: استخدام self._loop بدلاً من self.loop
    def invoke_later(self, fn):
        self._loop.call_soon_threadsafe(fn)

    # (إبقاء التعريف موحّدًا)
    def _training_complete(self):
        self.status_label.text = "Training complete."
        self.train_model_button.enabled = True
        self.save_model_button.enabled = True
        self.training_in_progress = False

    def _training_error(self, error_message):
        self.status_label.text = f"Training error: {error_message}"
        self.train_model_button.enabled = True
        self.training_in_progress = False

    def save_model(self, widget):
        asyncio.ensure_future(self._save_model_async())

    async def _save_model_async(self):
        try:
            dialog = toga.SaveFileDialog(
                title="Save Model",
                suggested_filename="cnn_model.h5",
                file_types=["h5", "hdf5"],
            )
            save_path = await self.main_window.dialog(dialog)
            if not save_path:                 # user pressed Cancel
                return

            # actual write
            self.model_handler.save_model(save_path)
            self.current_model_path = save_path

            await self.main_window.dialog(
                toga.InfoDialog("Model saved", f"Model stored at:\n{save_path}")
            )

        except Exception as exc:
            await self.main_window.dialog(
                toga.ErrorDialog("Error", f"Error saving model:\n{exc}")
            )

    def open_evaluation_window(self, widget):
        """Open the model evaluation window"""
        self.evaluation_window = toga.Window(title="Model Evaluation")

        # Create a box for evaluation options
        eval_box = toga.Box(style=Pack(direction=COLUMN, padding=10))

        # Title
        title_label = toga.Label(
            "CNN Model Evaluation",
            style=Pack(padding=(0, 0, 20, 0), font_size=16)
        )
        eval_box.add(title_label)

        # Model selection
        model_label = toga.Label("CNN Model:")
        self.model_input = toga.TextInput(readonly=True)
        model_button = toga.Button(
            "Select Model File",
            on_press=self.select_model_file
        )
        model_box = toga.Box(style=Pack(direction=ROW, padding=5))
        model_box.add(model_label)
        model_box.add(self.model_input)
        model_box.add(model_button)
        eval_box.add(model_box)

        # Test data selection
        test_label = toga.Label("Test Data:")
        self.test_input = toga.TextInput(readonly=True)
        test_button = toga.Button(
            "Select Test Data",
            on_press=self.select_test_data
        )
        test_box = toga.Box(style=Pack(direction=ROW, padding=5))
        test_box.add(test_label)
        test_box.add(self.test_input)
        test_box.add(test_button)
        eval_box.add(test_box)

        mode_box = toga.Box(style=Pack(direction=ROW, padding=5))
        mode_box.add(toga.Label("Evaluation Mode:"))

        self.single_image_switch = toga.Switch("Single Image",  value=False)
        self.folder_switch       = toga.Switch("Test Folder (with classes)", value=True)

        def on_single_toggle(widget):
            if widget.value:
                self.folder_switch.value = False

        def on_folder_toggle(widget):
            if widget.value:
                self.single_image_switch.value = False

        self.single_image_switch.on_toggle = on_single_toggle
        self.folder_switch.on_toggle       = on_folder_toggle

        mode_box.add(self.single_image_switch)
        mode_box.add(self.folder_switch)
        eval_box.add(mode_box)

        self.eval_status_label = toga.Label(
            "Ready for evaluation.", style=Pack(padding=(5, 0, 10, 0))
        )
        eval_box.add(self.eval_status_label)

        # ── evaluate button ──────────────────────────────────────
        self.evaluate_button = toga.Button(
            "Evaluate Model",
            on_press=self.evaluate_model,
            style=Pack(padding=(20, 5)),
        )
        eval_box.add(self.evaluate_button)

        # Results section
        results_label = toga.Label(
            "Evaluation Results:",
            style=Pack(padding=(10, 0, 5, 0), font_size=14)
        )
        eval_box.add(results_label)

        # Box to display metrics
        self.metrics_box = toga.Box(style=Pack(direction=COLUMN, padding=5))

        # Accuracy
        accuracy_box = toga.Box(style=Pack(direction=ROW, padding=2))
        accuracy_label = toga.Label("Accuracy:")
        self.accuracy_value = toga.Label("--")
        accuracy_box.add(accuracy_label)
        accuracy_box.add(self.accuracy_value)
        self.metrics_box.add(accuracy_box)

        # Precision
        precision_box = toga.Box(style=Pack(direction=ROW, padding=2))
        precision_label = toga.Label("Precision:")
        self.precision_value = toga.Label("--")
        precision_box.add(precision_label)
        precision_box.add(self.precision_value)
        self.metrics_box.add(precision_box)

        # F1 Score
        f1_box = toga.Box(style=Pack(direction=ROW, padding=2))
        f1_label = toga.Label("F1 Score:")
        self.f1_value = toga.Label("--")
        f1_box.add(f1_label)
        f1_box.add(self.f1_value)
        self.metrics_box.add(f1_box)

        # Single image prediction result (hidden initially)
        self.prediction_box = toga.Box(style=Pack(direction=COLUMN, padding=5))
        prediction_label = toga.Label("Prediction:")
        self.prediction_value = toga.Label("--")
        confidence_label = toga.Label("Confidence:")
        self.confidence_value = toga.Label("--")
        self.prediction_box.add(prediction_label)
        self.prediction_box.add(self.prediction_value)
        self.prediction_box.add(confidence_label)
        self.prediction_box.add(self.confidence_value)

        eval_box.add(self.metrics_box)
        eval_box.add(self.prediction_box)
        self.prediction_box.style.visibility = "hidden"

        # Placeholder for confusion matrix visualization
        self.confusion_matrix_box = toga.Box(
            style=Pack(width=300, height=300, padding=5)
        )
        self.confusion_matrix_view = toga.ImageView(
            style=Pack(width=300, height=300)
        )
        self.confusion_matrix_box.add(self.confusion_matrix_view)
        eval_box.add(self.confusion_matrix_box)

        # Set the content of the evaluation window
        self.evaluation_window.content = eval_box

        # Show the evaluation window
        self.evaluation_window.show()

    def select_model_file(self, widget):
        async def _run():
            try:
                dialog = toga.OpenFileDialog(
                    title="Select Model File",
                    file_types=["h5", "hdf5"],
                )
                path = await self.main_window.dialog(dialog)
                if path:
                    self.model_input.value = path
            except Exception as e:
                await self.main_window.dialog(
                    toga.ErrorDialog("Error", f"Error selecting file:\n{e}")
                )

        asyncio.ensure_future(_run())

    def select_test_data(self, widget):
        async def _run():
            try:
                # choose single image or folder based on the *value* of the switches
                if self.single_image_switch.value:
                    dialog = toga.OpenFileDialog(
                        title="Select Test Image",
                        file_types=["jpg", "jpeg", "png"],
                    )
                    path = await self.main_window.dialog(dialog)
                else:
                    dialog = toga.SelectFolderDialog("Select Test Data Folder")
                    path = await self.main_window.dialog(dialog)

                if path:
                    self.test_input.value = path
            except Exception as e:
                await self.main_window.dialog(
                    toga.ErrorDialog("Error", f"Error selecting data:\n{e}")
                )

        asyncio.ensure_future(_run())

    def evaluate_model(self, widget):
        # basic validation
        if not self.model_input.value:
            self.main_window.error_dialog("Input Error", "Please select a model file")
            return

        if not self.test_input.value:
            self.main_window.error_dialog("Input Error", "Please select test data")
            return

        # lock UI & show first status
        self.evaluate_button.enabled = False
        self.eval_status_label.text = "Loading model…"

        # run evaluation in background
        threading.Thread(target=self._evaluate_model_thread, daemon=True).start()

    # ──────────────────────────────────────────────────────────────
    #  BACKGROUND THREAD
    # ──────────────────────────────────────────────────────────────
    def _evaluate_model_thread(self):
        try:
            print("Loading model…")
            # 1. load model
            self.invoke_later(lambda: self._set_eval_status("Loading model…"))
            self.model_handler.load_model(self.model_input.value)
            print("model loaded")

            # 2. run evaluation
            if self.single_image_switch.value:
                print("Evaluating single img")
                self.invoke_later(lambda: self._set_eval_status("Evaluating image…"))
                result = self.model_handler.evaluate_single_image(self.test_input.value)
                print("done")
                self.invoke_later(lambda r=result: self._display_single_image_result(r))
            else:
                print("Evaluating folder")
                self.invoke_later(lambda: self._set_eval_status("Evaluating test set…"))
                result = self.model_handler.evaluate_on_folder(self.test_input.value)
                print("done")
                self.invoke_later(lambda r=result: self._display_evaluation_result(r))

        except Exception as exc:
            self.invoke_later(lambda m=str(exc): self._evaluation_error(m))

    def _set_eval_status(self, msg: str):
        print("Status: ", msg)
        self.eval_status_label.text = msg

    def _display_single_image_result(self, result):
        self._set_eval_status("Done.")

        # Hide metrics box and show prediction box
        self.metrics_box.style.visibility = "hidden"
        self.prediction_box.style.visibility = "visible"
        self.confusion_matrix_box.style.visibility = "hidden"

        # Update prediction values
        self.prediction_value.text = result['class_name']
        self.confidence_value.text = f"{result['confidence'] * 100:.2f}%"

        self.evaluate_button.enabled = True

    def _display_evaluation_result(self, result):
        self._set_eval_status("Done.")

        # Hide prediction box and show metrics box
        self.metrics_box.style.visibility = "visible"
        self.prediction_box.style.visibility = "hidden"
        self.confusion_matrix_box.style.visibility = "visible"

        # Update metric values
        self.accuracy_value.text = f"{result['accuracy'] * 100:.2f}%"
        self.precision_value.text = f"{result['precision'] * 100:.2f}%"
        self.f1_value.text = f"{result['f1_score'] * 100:.2f}%"

        # Convert confusion matrix image to a Toga Image
        if 'confusion_matrix_figure' in result:
            # Get the bytes from the confusion matrix figure
            cm_bytes = result['confusion_matrix_figure'].getvalue()

            # Create a temporary file to store the image
            temp_path = os.path.join(os.path.expanduser('~'), 'temp_cm.png')
            with open(temp_path, 'wb') as f:
                f.write(cm_bytes)

            # Load the image into the ImageView
            self.confusion_matrix_view.image = toga.Image(temp_path)

            # Clean up the temporary file (optional, can be done in a background task)
            try:
                os.remove(temp_path)
            except:
                pass

        self.evaluate_button.enabled = True

    def _evaluation_error(self, error_message):
        self._set_eval_status("Error")

        self.main_window.error_dialog(
            "Evaluation Error",
            f"An error occurred during evaluation: {error_message}"
        )

        self.evaluate_button.enabled = True


def main():
    return CNNTrainerApp("CNN Trainer", "org.example.cnntrainer")

if __name__ == "__main__":
    app = main()
    app.main_loop()
