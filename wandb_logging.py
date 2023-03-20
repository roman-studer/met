from sklearn.metrics import confusion_matrix
import wandb
import config
import uuid


class WandBLogger:
    def __init__(self):
        c = config.WandBConfig()
        c.run_name = str(uuid.uuid4())
        self.wandb = wandb
        self.wandb.init(project=c.project_name, entity=c.entity, name=c.run_name, sync_tensorboard=c.sync_tensorboard)
        self.run_name = self.wandb.run.name

    @staticmethod
    def log_confusion_matrix(y_true, y_pred, class_names) -> None:
        """
        Log a confusion matrix to wandb
        :param y_true: true labels
        :param y_pred: predicted labels
        :param class_names: list of class names
        :return: None
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Log the confusion matrix as an artifact
        table = wandb.Table(data=cm, columns=class_names, rows=class_names)
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred,
                                                                   class_names=class_names)})

    def log_metrics(self, metrics, step):
        """
        Log metrics to wandb
        :param metrics: dictionary of metrics
        :param step: step number
        :return: None
        """
        self.wandb.log(metrics, step=step)

    def log_config(self, config: dict):
        """
        Log configurations to wandb
        :param config: dict, configurations to log
        :return: None
        """
        self.wandb.config.update(config)

    def log_model(self, model, step):
        """
        Log model to wandb
        :param model: model to log
        :param step: step number
        :return: None
        """
        self.wandb.log({"model": model}, step=step)

    def log_artifact(self, artifact, name):
        """
        Log artifact to wandb
        :param artifact: artifact to log
        :param name: name of artifact
        :return: None
        """
        self.wandb.log_artifact(artifact, name=name)

    def log_image(self, image, name):
        """
        Log image to wandb
        :param image: image to log
        :param name: name of image
        :return: None
        """
        self.wandb.log({name: [wandb.Image(image)]})

    def log_text(self, text, name):
        """
        Log text to wandb
        :param text: text to log
        :param name: name of text
        :return: None
        """
        self.wandb.log({name: wandb.Html(text)})

    def finish(self):
        """
        Finish logging
        :return: None
        """
        self.wandb.finish()
