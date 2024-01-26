"""Contrast losses for incremental learning."""
from typing import Optional
import torch
import torch.nn.functional as F
from ilmot.model.losses.base import BaseLoss  # type:ignore


class MeanContrastLoss(BaseLoss):  # type:ignore
    """Feature distillation loss."""

    def __init__(
        self,
        weight_pull: float = 0.1,
        weight_push: float = 0.1,
        push_std: float = 3,
        pull_margin: float = 3,
    ):
        """Init."""
        super().__init__()
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.push_std = push_std
        self.pull_margin = pull_margin

    def __call__(  # pylint: disable=arguments-differ
        self,
        embeddings: torch.Tensor,  # N, 1024
        class_ids: torch.Tensor,  # N
        means: torch.Tensor,  # num_classes, 1024
        stds: torch.Tensor,
        class_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass."""
        num_classes, _ = means.shape
        mean_class_ids, _ = torch.sort(torch.unique(class_ids))

        # calculating the pulling loss
        true_means = means[class_ids, :]  # N, 1024
        pull_dists = torch.linalg.norm(embeddings - true_means, dim=-1)  # N

        # hard coded
        pull_dists = torch.clamp(pull_dists - self.pull_margin, min=0)  # N
        pull_dists = pull_dists ** 2  # l2
        loss_pull = []
        for cat_id in mean_class_ids:
            # average over class instances
            loss_pull.append(torch.mean(pull_dists[class_ids == cat_id]))
        # average over classes
        loss_pull = sum(loss_pull) / len(loss_pull)  # type: ignore

        mean_embeddings = []
        for cat_id in mean_class_ids:
            mean_embeddings.append(
                torch.mean(embeddings[class_ids == cat_id, :], dim=0)
            )
        mean_embeddings = torch.stack(mean_embeddings, dim=0)
        std_embeddings = stds[mean_class_ids, :]
        M, _ = mean_embeddings.shape  # type: ignore # pylint: disable=invalid-name

        if class_mask is None:
            class_mask = torch.ones(
                num_classes, dtype=torch.bool, device=embeddings.device
            )
        mean_embeddings = mean_embeddings.unsqueeze(1).repeat(  # type: ignore
            1, num_classes, 1
        )
        std_embeddings = std_embeddings.unsqueeze(1).repeat(1, num_classes, 1)
        mean_embeddings = mean_embeddings[:, class_mask, :]  # type: ignore
        std_embeddings = std_embeddings[:, class_mask, :]
        means = means[class_mask, :]
        means = means.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        stds = stds[class_mask, :]
        stds = stds.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        # prevent overflow
        dists = torch.abs(mean_embeddings - means)
        dists_normalized = dists / (
            stds + std_embeddings + 1e-5
        )  # M , num_classes, 1024

        # nomalize with means
        mean_class_ids[mean_class_ids < 0] = (
            mean_class_ids[mean_class_ids < 0] + num_classes
        )
        pos_mask = F.one_hot(
            mean_class_ids, num_classes=num_classes
        )  # N , num_classes

        # compute pushing loss
        pos_mask = pos_mask[:, class_mask]
        # use 3 std, only when std is greater than 0.01
        loss_push = torch.clamp(
            self.push_std - dists_normalized[pos_mask == 0, ...],
            min=0,
        )

        loss_push = torch.mean(loss_push ** 2)  # l2

        return loss_pull * self.weight_pull, loss_push * self.weight_push  # type: ignore # pylint: disable=line-too-long


class AdaptiveContrastLoss(BaseLoss):  # type:ignore
    """Feature distillation loss."""

    def __init__(
        self,
        weight_pull: float = 0.1,
        weight_push: float = 0.1,
        push_std: float = 3,
        pull_margin: float = 3,
    ):
        """Init."""
        super().__init__()
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.push_std = push_std
        self.pull_margin = pull_margin

    def __call__(  # pylint: disable=arguments-differ
        self,
        embeddings: torch.Tensor,  # N, 1024
        class_ids: torch.Tensor,  # N
        means: torch.Tensor,  # num_classes, 1024
        stds: torch.Tensor,
        class_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass."""
        num_classes, _ = means.shape
        mean_class_ids, _ = torch.sort(torch.unique(class_ids))

        # calculating the pulling loss
        true_means = means[class_ids, :]  # N, 1024
        pull_dists = torch.linalg.norm(embeddings - true_means, dim=-1)  # N

        # hard coded
        pull_dists = torch.clamp(pull_dists - self.pull_margin, min=0)  # N
        pull_dists = pull_dists ** 2  # l2
        loss_pull = []
        for cat_id in mean_class_ids:
            # average over class instances
            loss_pull.append(torch.mean(pull_dists[class_ids == cat_id]))
        # average over classes
        loss_pull = sum(loss_pull) / len(loss_pull)  # type: ignore

        mean_embeddings = []
        for cat_id in mean_class_ids:
            mean_embeddings.append(
                torch.mean(embeddings[class_ids == cat_id, :], dim=0)
            )
        mean_embeddings = torch.stack(mean_embeddings, dim=0)
        std_embeddings = stds[mean_class_ids, :]
        M, _ = mean_embeddings.shape  # type: ignore # pylint: disable=invalid-name

        if class_mask is None:
            class_mask = torch.ones(
                num_classes, dtype=torch.bool, device=embeddings.device
            )
        mean_embeddings = mean_embeddings.unsqueeze(1).repeat(  # type: ignore
            1, num_classes, 1
        )
        std_embeddings = std_embeddings.unsqueeze(1).repeat(1, num_classes, 1)
        mean_embeddings = mean_embeddings[:, class_mask, :]  # type: ignore
        std_embeddings = std_embeddings[:, class_mask, :]
        means = means[class_mask, :]
        means = means.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        stds = stds[class_mask, :]
        stds = stds.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        # prevent overflow
        diffs = mean_embeddings - means  # M , num_classes, 1024

        std_sum = (stds + std_embeddings + 1e-5) ** 2  # M , num_classes, 1024
        std_sum = 1.0 / std_sum

        dists = diffs * std_sum * diffs

        dists = torch.sqrt(torch.sum(dists, dim=-1))  # M , num_classes

        # nomalize with means
        mean_class_ids[mean_class_ids < 0] = (
            mean_class_ids[mean_class_ids < 0] + num_classes
        )
        pos_mask = F.one_hot(
            mean_class_ids, num_classes=num_classes
        )  # N , num_classes

        # compute pushing loss
        pos_mask = pos_mask[:, class_mask]
        # use 3 std, only when std is greater than 0.01
        loss_push = torch.clamp(
            self.push_std - dists[pos_mask == 0],
            min=0,
        )

        loss_push = torch.mean(loss_push ** 2)  # l2

        return loss_pull * self.weight_pull, loss_push * self.weight_push  # type: ignore # pylint: disable=line-too-long


class AdaptiveContrastLoss2(BaseLoss):  # type:ignore
    """Feature distillation loss."""

    def __init__(
        self,
        weight_pull: float = 0.1,
        weight_push: float = 0.1,
        push_std: float = 3,
        pull_margin: float = 3,
    ):
        """Init."""
        super().__init__()
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.push_std = push_std
        self.pull_margin = pull_margin

    def __call__(  # pylint: disable=arguments-differ
        self,
        embeddings: torch.Tensor,  # N, 1024
        class_ids: torch.Tensor,  # N
        means: torch.Tensor,  # num_classes, 1024
        stds: torch.Tensor,
        class_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass."""
        num_classes, _ = means.shape
        mean_class_ids, _ = torch.sort(torch.unique(class_ids))

        # calculating the pulling loss
        true_means = means[class_ids, :]  # N, 1024
        true_stds = stds[class_ids, :]
        true_stds = (true_stds + 1e-5) ** 2  # N, 1024
        true_stds = 1.0 / true_stds
        pull_diffs = embeddings - true_means
        pull_dists = pull_diffs * true_stds * pull_diffs
        pull_dists = torch.sqrt(torch.sum(pull_dists, dim=-1))  # N

        # hard coded
        pull_dists = torch.clamp(pull_dists - self.pull_margin, min=0)  # N
        pull_dists = pull_dists ** 2  # l2
        loss_pull = []
        for cat_id in mean_class_ids:
            # average over class instances
            loss_pull.append(torch.mean(pull_dists[class_ids == cat_id]))
        # average over classes
        loss_pull = sum(loss_pull) / len(loss_pull)  # type: ignore

        mean_embeddings = []
        for cat_id in mean_class_ids:
            mean_embeddings.append(
                torch.mean(embeddings[class_ids == cat_id, :], dim=0)
            )
        mean_embeddings = torch.stack(mean_embeddings, dim=0)
        std_embeddings = stds[mean_class_ids, :]
        M, _ = mean_embeddings.shape  # type: ignore # pylint: disable=invalid-name

        if class_mask is None:
            class_mask = torch.ones(
                num_classes, dtype=torch.bool, device=embeddings.device
            )
        mean_embeddings = mean_embeddings.unsqueeze(1).repeat(  # type: ignore
            1, num_classes, 1
        )
        std_embeddings = std_embeddings.unsqueeze(1).repeat(1, num_classes, 1)
        mean_embeddings = mean_embeddings[:, class_mask, :]  # type: ignore
        std_embeddings = std_embeddings[:, class_mask, :]
        means = means[class_mask, :]
        means = means.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        stds = stds[class_mask, :]
        stds = stds.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        # prevent overflow
        diffs = mean_embeddings - means  # M , num_classes, 1024

        std_sum = (stds + std_embeddings + 1e-5) ** 2  # M , num_classes, 1024
        std_sum = 1.0 / std_sum

        dists = diffs * std_sum * diffs

        dists = torch.sqrt(torch.sum(dists, dim=-1))  # M , num_classes

        # nomalize with means
        mean_class_ids[mean_class_ids < 0] = (
            mean_class_ids[mean_class_ids < 0] + num_classes
        )
        pos_mask = F.one_hot(
            mean_class_ids, num_classes=num_classes
        )  # N , num_classes

        # compute pushing loss
        pos_mask = pos_mask[:, class_mask]
        # use 3 std, only when std is greater than 0.01
        loss_push = torch.clamp(
            self.push_std - dists[pos_mask == 0],
            min=0,
        )

        loss_push = torch.mean(loss_push ** 2)  # l2

        return loss_pull * self.weight_pull, loss_push * self.weight_push  # type: ignore # pylint: disable=line-too-long


class MahaContrastLoss(BaseLoss):  # type:ignore
    """Feature distillation loss."""

    def __init__(
        self,
        weight_pull: float = 0.1,
        weight_push: float = 0.1,
        push_std: float = 3,
        pull_margin: float = 3,
    ):
        """Init."""
        super().__init__()
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.push_std = push_std
        self.pull_margin = pull_margin
        self.std_shrink = 0.5

    def __call__(  # pylint: disable=arguments-differ
        self,
        embeddings: torch.Tensor,  # N, 1024
        class_ids: torch.Tensor,  # N
        means: torch.Tensor,  # num_classes, 1024
        stds: torch.Tensor,
        class_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass."""
        num_classes, _ = means.shape
        mean_class_ids, _ = torch.sort(torch.unique(class_ids))

        # calculating the pulling loss
        true_means = means[class_ids, :]  # N, 1024
        true_stds = stds[class_ids, :]
        pull_diffs = embeddings - true_means
        pull_dists = torch.linalg.norm(pull_diffs, dim=-1)  # N
        true_stds = torch.linalg.norm(true_stds, dim=-1)  # N

        # hard coded
        pull_dists = pull_dists ** 2  # l2
        loss_pull = []
        for cat_id in mean_class_ids:
            # average over class instances
            if torch.linalg.norm(stds[cat_id]) > self.pull_margin:
                loss_pull.append(torch.mean(pull_dists[class_ids == cat_id]))
        # average over classes
        if len(loss_pull) > 0:
            loss_pull = sum(loss_pull) / len(loss_pull)  # type: ignore
        else:
            loss_pull = torch.zeros(1, dtype=means.dtype, device=means.device)

        mean_embeddings = []
        for cat_id in mean_class_ids:
            mean_embeddings.append(
                torch.mean(embeddings[class_ids == cat_id, :], dim=0)
            )
        mean_embeddings = torch.stack(mean_embeddings, dim=0)
        std_embeddings = stds[mean_class_ids, :]
        M, _ = mean_embeddings.shape  # type: ignore # pylint: disable=invalid-name

        if class_mask is None:
            class_mask = torch.ones(
                num_classes, dtype=torch.bool, device=embeddings.device
            )
        mean_embeddings = mean_embeddings.unsqueeze(1).repeat(  # type: ignore
            1, num_classes, 1
        )
        std_embeddings = std_embeddings.unsqueeze(1).repeat(1, num_classes, 1)
        mean_embeddings = mean_embeddings[:, class_mask, :]  # type: ignore
        std_embeddings = std_embeddings[:, class_mask, :]
        means = means[class_mask, :]
        means = means.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        stds = stds[class_mask, :]
        stds = stds.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        # prevent overflow
        diffs = mean_embeddings - means  # M , num_classes, 1024

        std_sum = (
            stds ** 2 + std_embeddings ** 2 + 1e-5
        ) / 2  # M , num_classes, 1024
        std_sum = 1.0 / std_sum

        dists = diffs * std_sum * diffs

        dists = torch.sqrt(torch.sum(dists, dim=-1))  # M , num_classes

        # nomalize with means
        mean_class_ids[mean_class_ids < 0] = (
            mean_class_ids[mean_class_ids < 0] + num_classes
        )
        pos_mask = F.one_hot(
            mean_class_ids, num_classes=num_classes
        )  # N , num_classes

        # compute pushing loss
        pos_mask = pos_mask[:, class_mask]
        # use 3 std, only when std is greater than 0.01
        loss_push = torch.clamp(
            self.push_std - dists[pos_mask == 0],
            min=0,
        )

        loss_push = torch.mean(loss_push ** 2)  # l2

        return loss_pull * self.weight_pull, loss_push * self.weight_push  # type: ignore # pylint: disable=line-too-long


class MahaContrastLossNew(BaseLoss):  # type:ignore
    """Feature distillation loss."""

    def __init__(
        self,
        weight_pull: float = 0.1,
        weight_push: float = 0.1,
        push_std: float = 3,
        pull_margin: float = 3,
    ):
        """Init."""
        super().__init__()
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.push_std = push_std
        self.pull_margin = pull_margin

    def __call__(  # pylint: disable=arguments-differ
        self,
        embeddings: torch.Tensor,  # N, 1024
        class_ids: torch.Tensor,  # N
        means: torch.Tensor,  # num_classes, 1024
        stds: torch.Tensor,
        class_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass."""
        num_classes, _ = means.shape
        mean_class_ids, _ = torch.sort(torch.unique(class_ids))

        # calculating the pulling loss
        true_means = means[class_ids, :]  # N, 1024
        pull_diffs = embeddings - true_means

        pull_diffs = pull_diffs ** 2
        # estimate current variance
        loss_pull = []
        for cat_id in mean_class_ids:
            estimated_std = torch.sqrt(
                torch.mean(pull_diffs[class_ids == cat_id], dim=0)
            )  # 512
            loss_pull.append(
                torch.sum((estimated_std - self.pull_margin) ** 2)
            )
        # average over classes
        loss_pull = sum(loss_pull) / len(loss_pull)  # type: ignore

        mean_embeddings = []
        for cat_id in mean_class_ids:
            mean_embeddings.append(
                torch.mean(embeddings[class_ids == cat_id, :], dim=0)
            )
        mean_embeddings = torch.stack(mean_embeddings, dim=0)
        std_embeddings = stds[mean_class_ids, :]
        M, _ = mean_embeddings.shape  # type: ignore # pylint: disable=invalid-name

        if class_mask is None:
            class_mask = torch.ones(
                num_classes, dtype=torch.bool, device=embeddings.device
            )
        mean_embeddings = mean_embeddings.unsqueeze(1).repeat(  # type: ignore
            1, num_classes, 1
        )
        std_embeddings = std_embeddings.unsqueeze(1).repeat(1, num_classes, 1)
        mean_embeddings = mean_embeddings[:, class_mask, :]  # type: ignore
        std_embeddings = std_embeddings[:, class_mask, :]
        means = means[class_mask, :]
        means = means.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        stds = stds[class_mask, :]
        stds = stds.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        # prevent overflow
        diffs = mean_embeddings - means  # M , num_classes, 1024

        std_sum = (
            stds ** 2 + std_embeddings ** 2 + 1e-5
        ) / 2  # M , num_classes, 1024
        std_sum = 1.0 / std_sum

        dists = diffs * std_sum * diffs

        dists = torch.sqrt(torch.sum(dists, dim=-1))  # M , num_classes

        # nomalize with means
        mean_class_ids[mean_class_ids < 0] = (
            mean_class_ids[mean_class_ids < 0] + num_classes
        )
        pos_mask = F.one_hot(
            mean_class_ids, num_classes=num_classes
        )  # N , num_classes

        # compute pushing loss
        pos_mask = pos_mask[:, class_mask]
        # use 3 std, only when std is greater than 0.01
        loss_push = torch.clamp(
            self.push_std - dists[pos_mask == 0],
            min=0,
        )

        loss_push = torch.mean(loss_push ** 2)  # l2

        return loss_pull * self.weight_pull, loss_push * self.weight_push  # type: ignore # pylint: disable=line-too-long


class MahaContrastLossLog(BaseLoss):  # type:ignore
    """Feature distillation loss."""

    def __init__(
        self,
        weight_pull: float = 0.1,
        weight_push: float = 0.1,
        push_std: float = 3,
        pull_margin: float = 3,
    ):
        """Init."""
        super().__init__()
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.push_std = push_std
        self.pull_margin = pull_margin

    def __call__(  # pylint: disable=arguments-differ
        self,
        embeddings: torch.Tensor,  # N, 1024
        class_ids: torch.Tensor,  # N
        means: torch.Tensor,  # num_classes, 1024
        stds: torch.Tensor,
        class_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass."""
        num_classes, _ = means.shape
        mean_class_ids, _ = torch.sort(torch.unique(class_ids))

        # calculating the pulling loss
        true_means = means[class_ids, :]  # N, 1024
        pull_diffs = embeddings - true_means

        pull_diffs = pull_diffs ** 2
        # estimate current variance
        loss_pull = []
        for cat_id in mean_class_ids:
            estimated_std = torch.sqrt(
                torch.mean(pull_diffs[class_ids == cat_id], dim=0)
            )  # 512
            loss_per_dim = torch.log(
                (estimated_std ** 2 + self.pull_margin ** 2 + 1e-6)
                / (2 * estimated_std * self.pull_margin + 1e-6)
            )
            loss_pull.append(torch.sum(loss_per_dim))
        # average over classes
        loss_pull = sum(loss_pull) / len(loss_pull)  # type: ignore

        mean_embeddings = []
        for cat_id in mean_class_ids:
            mean_embeddings.append(
                torch.mean(embeddings[class_ids == cat_id, :], dim=0)
            )
        mean_embeddings = torch.stack(mean_embeddings, dim=0)
        std_embeddings = stds[mean_class_ids, :]
        M, _ = mean_embeddings.shape  # type: ignore # pylint: disable=invalid-name

        if class_mask is None:
            class_mask = torch.ones(
                num_classes, dtype=torch.bool, device=embeddings.device
            )
        mean_embeddings = mean_embeddings.unsqueeze(1).repeat(  # type: ignore
            1, num_classes, 1
        )
        std_embeddings = std_embeddings.unsqueeze(1).repeat(1, num_classes, 1)
        mean_embeddings = mean_embeddings[:, class_mask, :]  # type: ignore
        std_embeddings = std_embeddings[:, class_mask, :]
        means = means[class_mask, :]
        means = means.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        stds = stds[class_mask, :]
        stds = stds.unsqueeze(0).repeat(M, 1, 1)  # M , num_classes, 1024
        # prevent overflow
        diffs = mean_embeddings - means  # M , num_classes, 1024

        std_sum = (
            stds ** 2 + std_embeddings ** 2 + 1e-6
        ) / 2  # M , num_classes, 1024
        std_sum = 1.0 / std_sum

        dists = diffs * std_sum * diffs

        dists = torch.sqrt(torch.sum(dists, dim=-1))  # M , num_classes

        # nomalize with means
        mean_class_ids[mean_class_ids < 0] = (
            mean_class_ids[mean_class_ids < 0] + num_classes
        )
        pos_mask = F.one_hot(
            mean_class_ids, num_classes=num_classes
        )  # N , num_classes

        # compute pushing loss
        pos_mask = pos_mask[:, class_mask]
        # use 3 std, only when std is greater than 0.01
        loss_push = torch.clamp(
            self.push_std - dists[pos_mask == 0],
            min=0,
        )

        loss_push = torch.mean(loss_push ** 2)  # l2

        return loss_pull * self.weight_pull, loss_push * self.weight_push  # type: ignore # pylint: disable=line-too-long


class HingeContrastLoss(BaseLoss):  # type:ignore
    """Feature distillation loss."""

    def __init__(self, weight: float = 0.1, margin=15):
        """Init."""
        super().__init__()
        self.weight = weight
        self.hingeloss = torch.nn.HingeEmbeddingLoss(
            margin=margin, reduction="none"
        )

    def __call__(  # pylint: disable=arguments-differ
        self,
        embeddings: torch.Tensor,  # N, 1024
        class_ids: torch.Tensor,  # N
        means: torch.Tensor,  # num_classes, 1024
        stds: torch.Tensor,
        class_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass."""
        num_classes, _ = means.shape
        if class_mask is None:
            class_mask = torch.ones(
                num_classes, dtype=torch.bool, device=embeddings.device
            )

        distances = torch.cdist(embeddings, means)
        labels = torch.zeros_like(
            distances, dtype=torch.long, device=distances.device
        )
        labels[...] = -1
        for i, cat_id in enumerate(class_ids):
            labels[i, cat_id] = 1

        distances = distances[:, class_mask]
        labels = labels[:, class_mask]

        loss = self.hingeloss(distances, labels)

        loss_pull = loss[labels == 1].mean()
        loss_push = loss[labels == -1].mean()

        return loss_pull * self.weight, loss_push * self.weight  # type: ignore
