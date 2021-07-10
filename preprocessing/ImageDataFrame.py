from pathlib import Path
from typing import Optional
import pandas as pd


class ImageDataFrame:
    """
        Wrapper around a dataframe to provide consistent data passing between functions.
        The DataFrame can be accessed via the frame variable.
        Columns: ["stem", "img_file", "label_file", "is_test", "class"]
    """
    frame: pd.DataFrame

    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        if isinstance(df, pd.DataFrame):
            self.frame = df
        else:
            self.frame = pd.DataFrame(
                columns=["stem", "img_file", "label_file", "is_test", "class"])

    def addImg(self, img_path: Path, label_path: Path, img_class: str, is_test: Optional[bool] = None):
        """Add an image to the dataframe. Img and txt path are converted to absolutes

        Args:
            img_path (Path): Path to the image
            txt_path (Path): The path to the label
            img_class (str): The image class
            is_test (Optional[bool], optional): Optional is_test boolean. Defaults to None.
        """
        row = [img_path.stem, img_path.absolute(), label_path.absolute(), is_test,
               img_class]
        self.frame.loc[len(self.frame)] = row
