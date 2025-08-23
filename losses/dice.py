import smp


class NamedDiceLoss(smp.losses.DiceLoss):
    @property
    def __name__(self):
        return "dice_loss"