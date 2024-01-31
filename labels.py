from enum import Enum

class Labels(Enum):
    CLICK = 0
    CURSOR = 1
    FIST = 2
    FLAT = 3
    GUN = 4
    INWARD = 5
    OPENAWAY = 6
    OPENFACING = 7
    OUTWARD = 8
    THUMBSUP = 9
    THUMBSDOWN = 10

LabelToStringMap = {
    Labels.CLICK : "Click",
    Labels.CURSOR : "Cursor",
    Labels.FIST : "Fist",
    Labels.FLAT : "Flat",
    Labels.GUN : "Gun",
    Labels.INWARD : "Inward",
    Labels.OPENAWAY : "Open Away",
    Labels.OPENFACING : "Open Facing",
    Labels.OUTWARD : "Outward",
    Labels.THUMBSUP : "Thumbs Up",
    Labels.THUMBSDOWN : "Thumbs Down"
}

LabelToImgPath = {
    Labels.CLICK : "./img/Click.png",
    Labels.CURSOR : "./img/Cursor.png",
    Labels.FIST : "./img/Fist.png",
    Labels.FLAT : "./img/Flat.png",
    Labels.GUN : "./img/Gun.png",
    Labels.INWARD : "./img/Inward.png",
    Labels.OPENAWAY : "./img/OpenAway.png",
    Labels.OPENFACING : "./img/OpenFacing.png",
    Labels.OUTWARD : "./img/Outward.png",
    Labels.THUMBSUP : "./img/ThumbsUp.png",
    Labels.THUMBSDOWN : "./img/ThumbsDown.png"
}