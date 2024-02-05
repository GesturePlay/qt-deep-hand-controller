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
    Labels.CLICK : "./Icons/Click.png",
    Labels.CURSOR : "./Icons/Cursor.png",
    Labels.FIST : "./Icons/Fist.png",
    Labels.FLAT : "./Icons/Flat.png",
    Labels.GUN : "./Icons/Gun.png",
    Labels.INWARD : "./Icons/Inward.png",
    Labels.OPENAWAY : "./Icons/OpenAway.png",
    Labels.OPENFACING : "./Icons/OpenFacing.png",
    Labels.OUTWARD : "./Icons/Outward.png",
    Labels.THUMBSUP : "./Icons/ThumbsUp.png",
    Labels.THUMBSDOWN : "./Icons/ThumbsDown.png"
}