import os
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, applications
from sklearn.metrics import precision_score, f1_score, confusion_matrix

AUTOTUNE      = tf.data.AUTOTUNE          
NUM_WORKERS   = tf.data.AUTOTUNE
CLASS_FILE = "classes.txt"

class CNNModelHandler:
    def __init__(self):
        self.model   = None
        self.trained = False
        self.history = None
        self.classes = None

    def create_custom_model(
        self,
        input_shape=(224, 224, 3),
        num_classes=10,
        conv_layers=3,
        pooling_layers=3,
    ):
        
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        if pooling_layers > 0:
            model.add(layers.MaxPooling2D((2, 2)))

        filters = 64
        for i in range(1, conv_layers):
            model.add(layers.Conv2D(filters, (3, 3), activation="relu"))
            if i < pooling_layers:
                model.add(layers.MaxPooling2D((2, 2)))
            filters = min(filters * 2, 512)

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation="softmax"))

        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        self.model = model
        return model

    def create_pretrained_model(
        self,
        model_name,
        input_shape=(224, 224, 3),
        num_classes=10,
    ):
        
        if model_name == "VGG16":
            base = applications.VGG16(weights="imagenet",
                                      include_top=False,
                                      input_shape=input_shape)
        elif model_name == "VGG19":
            base = applications.VGG19(weights="imagenet",
                                      include_top=False,
                                      input_shape=input_shape)
        elif model_name == "ResNet50":
            base = applications.ResNet50(weights="imagenet",
                                         include_top=False,
                                         input_shape=input_shape)
        elif model_name == "GoogLeNet/Inception":
            base = applications.InceptionV3(weights="imagenet",
                                            include_top=False,
                                            input_shape=input_shape)
        else:
            
            return self.create_custom_model(input_shape, num_classes)

        base.trainable = False

        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ])

        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        self.model = model
        return model

    # ------------------------------------------------------------------ #
    #  TRAIN
    # ------------------------------------------------------------------ #
    def train_model(
        self,
        data_path,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        status_callback=None,              
    ):
        
        if status_callback:
            status_callback("Listing files…")

        train_ds_raw = tf.keras.utils.image_dataset_from_directory(
            data_path,
            labels="inferred",
            label_mode="categorical",
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=True,
            follow_links=False,
        )

        val_ds_raw = tf.keras.utils.image_dataset_from_directory(
            data_path,
            labels="inferred",
            label_mode="categorical",
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=False,
            follow_links=False,
        )

        self.classes = train_ds_raw.class_names

        def configure(ds, training):
            ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                        num_parallel_calls=NUM_WORKERS)
            if training:
                ds = ds.cache().shuffle(1000)
            return ds.prefetch(AUTOTUNE)

        train_ds = configure(train_ds_raw, training=True)
        val_ds   = configure(val_ds_raw,   training=False)

        

        if status_callback:
            status_callback("Starting training…")

        
        if status_callback:
            class Relay(tf.keras.callbacks.Callback):
                def on_epoch_begin(s, epoch, logs=None):
                    status_callback(f"Epoch {epoch+1}/{epochs}…")
                def on_epoch_end(s, epoch, logs=None):
                    acc = logs.get("accuracy", 0)
                    status_callback(f"Epoch {epoch+1} finished  acc={acc:.3f}")

            cb = Relay()
            callbacks = [cb]
        else:
            callbacks = None
        

        self.history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
        )
        self.trained = True
        return self.history

    # ------------------------------------------------------------------ #
    #  SAVE / LOAD
    # ------------------------------------------------------------------ #
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(filepath)

        if self.classes:
            cls_path = os.path.join(os.path.dirname(filepath), CLASS_FILE)
            with open(cls_path, "w", encoding="utf8") as f:
                f.write("\n".join(self.classes))
            print(f"[DEBUG] saved {len(self.classes)} labels ➜ {cls_path}", flush=True)


    def load_model(self, filepath):
        self.model = models.load_model(filepath)
        self.trained = True
        self.input_size = self.model.input_shape[1:3]

        cls_path = os.path.join(os.path.dirname(filepath), CLASS_FILE)
        if os.path.exists(cls_path):
            with open(cls_path, encoding="utf8") as f:
                self.classes = [ln.strip() for ln in f if ln.strip()]
            print(f"[DEBUG] loaded {len(self.classes)} labels from {cls_path}", flush=True)
        else:
            self.classes = None
            print(f"[WARN] {CLASS_FILE} not found next to model – predictions will be 'Unknown'", flush=True)

        return self.model

    # ------------------------------------------------------------------ #
    #  EVALUATION HELPERS
    # ------------------------------------------------------------------ #
    def evaluate_on_folder(self, folder):

        if self.classes is None:
            raise RuntimeError("This model has no class list – retrain or add classes.txt")

        if self.model is None:
            raise ValueError("No model loaded")

        H, W = self.input_size
        print(f"[DEBUG] evaluate_on_folder → resizing to {H}×{W}", flush=True)

        num_out = self.model.output_shape[-1]
        label_mode = "binary" if num_out == 1 else "categorical"
        print("Model classification mode: ", label_mode)

        test_ds = tf.keras.utils.image_dataset_from_directory(
            folder,
            labels="inferred",
            label_mode=label_mode,
            shuffle=False,
            image_size=(H, W),
            batch_size=32,
        )

        
        if label_mode == "binary":
            # y_true comes as float32 0/1, shape (batch, 1)
            y_true = np.concatenate(list(test_ds.map(lambda x, y: y).as_numpy_iterator())).reshape(-1).astype(int)

            # logits → probabilities → 0/1 predictions
            probs  = self.model.predict(test_ds).reshape(-1)
            y_pred = (probs >= 0.5).astype(int)
        else:
            y_true = np.concatenate(list(test_ds.map(lambda x, y: y).as_numpy_iterator()))
            y_true = np.argmax(y_true, axis=1)

            probs  = self.model.predict(test_ds)
            y_pred = np.argmax(probs, axis=1)


        results     = self.model.evaluate(test_ds, verbose=0)
        predictions = self.model.predict(test_ds)
        y_pred      = np.argmax(predictions, axis=1)

        precision = precision_score(y_true, y_pred, average="weighted")
        f1        = f1_score(y_true, y_pred, average="weighted")
        cm        = confusion_matrix(y_true, y_pred)
        fig_buf   = self._plot_confusion_matrix(cm, self.classes)

        return dict(accuracy=results[1],
                    precision=precision,
                    f1_score=f1,
                    confusion_matrix=cm,
                    confusion_matrix_figure=fig_buf)

    def evaluate_single_image(self, image_path):

        if self.classes is None:
            raise RuntimeError("This model has no class list – retrain or add classes.txt")

        if self.model is None:
            raise ValueError("No model loaded")

        H, W = self.input_size
        print(f"[DEBUG] evaluate_single_image → resizing to {H}×{W}", flush=True)


        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(H, W))
        arr = tf.keras.preprocessing.image.img_to_array(img)[None, ...] / 255.0
        probs = self.model.predict(arr)[0]
        idx = int(np.argmax(probs))
        name = self.classes[idx] if self.classes and idx < len(self.classes) else "Unknown"
        return dict(class_index=idx,
                    class_name=name,
                    confidence=float(probs[idx]),
                    all_probabilities=probs.tolist())

    # ------------------------------------------------------------------ #
    #  UTIL
    # ------------------------------------------------------------------ #
    def _plot_confusion_matrix(self, cm, class_names):
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick = np.arange(len(class_names))
        plt.xticks(tick, class_names, rotation=45)
        plt.yticks(tick, class_names)

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return buf
