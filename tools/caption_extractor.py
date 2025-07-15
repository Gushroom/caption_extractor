import pandas as pd
import json
from sklearn.cluster import DBSCAN
import numpy as np

from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

class CaptionExtractorTool(Tool):
    def extract_caption(self, input_text: str) -> dict[str, str]: 
        try:
            data = json.loads(input_text)

            # Extract into df
            all_texts_df = pd.DataFrame()
            for item in data:
                df_normalized = pd.json_normalize(item['texts'])
                df_normalized['frame_index'] = item['index']
                all_texts_df = pd.concat([all_texts_df, df_normalized], ignore_index=True)

            all_texts_df = all_texts_df.rename(columns={
                'location.widthInPixel': 'width',
                'location.heightInPixel': 'height',
                'location.topOffsetInPixel': 'top',
                'location.leftOffsetInPixel': 'left'
            })

            # find centers coords
            all_texts_df['center_x'] = all_texts_df['left'] + all_texts_df['width'] / 2
            all_texts_df['center_y'] = all_texts_df['top'] + all_texts_df['height'] / 2

            # print(all_texts_df.head())

            max_y = (all_texts_df['top'] + all_texts_df['height']).max()
            lower_region_df = all_texts_df[all_texts_df['center_y'] > max_y * 0.4].copy()

            # print(lower_region_df.head())

            # DBSCAN 聚类
            coords = lower_region_df[['center_x', 'center_y']].values
            # eps 可以调整，大概约为字幕框高度即可
            # min_samples 越大越难判定为聚类
            clustering = DBSCAN(eps=50, min_samples=2).fit(coords)
            lower_region_df['cluster'] = clustering.labels_

            # 去噪
            clustered_df = lower_region_df[lower_region_df['cluster'] != -1]

            # 选择频率最高的块（字幕一般在下半区最频繁）
            if not clustered_df.empty:
                cluster_sizes = clustered_df['cluster'].value_counts()
                largest_cluster_label = cluster_sizes.idxmax()
                captions_df = clustered_df[clustered_df['cluster'] == largest_cluster_label].copy()

            # 去重
            # 若连续出现相同字幕，则合并进第一次出现的帧
            # 最多连续合三帧
            # 保持大概字字幕框位置
            captions_df['pos_bucket'] = list(zip(captions_df['top'] // 10, captions_df['left'] // 10))

            # 按帧排序
            captions_df.sort_values(by=['text', 'frame_index'], inplace=True)
            frame_diff = captions_df.groupby(['pos_bucket', 'text'])['frame_index'].diff()
            captions_df['block_id'] = (frame_diff != 1).cumsum()

            # 去重 合并
            is_first_in_block = captions_df.groupby('block_id').cumcount() == 0
            deduplicated_df = captions_df[is_first_in_block].copy()

            output_data = []
            grouped = deduplicated_df.groupby('frame_index')

            for frame_idx, group in grouped:
                texts_list = []
                for _, row in group.iterrows():
                    location_dict = {
                        "widthInPixel": int(row['width']),
                        "heightInPixel": int(row['height']),
                        "topOffsetInPixel": int(row['top']),
                        "leftOffsetInPixel": int(row['left'])
                    }
                    text_dict = {
                        "text": row['text'],
                        "location": location_dict
                    }
                    texts_list.append(text_dict)
                
                if texts_list:
                    frame_object = {
                        "index": int(frame_idx),
                        "texts": texts_list
                    }
                    output_data.append(frame_object)

            result = json.dumps(output_data, ensure_ascii=False, indent=4)
            return result
        except Exception as e:
            raise ValueError(f"Caption extraction failed: {str(e)}")


    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:

        raw_ocr_text = tool_parameters.get("raw_ocr_text")
        if not raw_ocr_text:
            yield self.create_json_message({
                "result": "请提供原始的OCR结果"
            })
            return
        result = self.extract_caption(raw_ocr_text)
        yield self.create_json_message({
            "result": result
        })
