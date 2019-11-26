import jieba


class Document:
    def __init__(self, original_path):
        self.original_path = original_path
        # self.destination_path = destination_path

    def segment(self):
        """
        Get segmentation of this paragraph.
        :return: String of segmentation aligned with "/"
        """
        with open(self.original_path, 'r') as f:
            text = f.read()
            f.close()
        seg_list = jieba.cut(text)
        return "/".join(seg_list)

