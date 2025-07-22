import json
import math
import re
from collections import Counter, defaultdict
from itertools import groupby
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class CaptionExtractorTool(Tool):
    def _clean_text(self, text: str) -> str:
        """
        Cleans a string, keeping only Chinese characters and numbers.

        Args:
            text: The input string.

        Returns:
            A new string containing only valid characters, or an empty string.
        """
        if not text:
            return ""
        pattern = re.compile(r'[\x00-\x2F\x3A-\x7F]')
        return pattern.sub('', text)
    def _custom_dbscan(self, coords: list[list[float]], eps: float, min_samples: int) -> list[int]:
        """
        A pure Python implementation of the DBSCAN clustering algorithm.

        Args:
            coords: A list of [x, y] coordinates for each text box.
            eps: The maximum distance between two samples for one to be considered
                 as in the neighborhood of the other.
            min_samples: The number of samples in a neighborhood for a point
                         to be considered as a core point.

        Returns:
            A list of cluster labels for each coordinate. Noise points are labeled -1.
        """
        NOISE = -1
        UNCLASSIFIED = 0
        
        labels = [UNCLASSIFIED] * len(coords)
        cluster_id = 0

        for i in range(len(coords)):
            # Skip already classified points
            if labels[i] != UNCLASSIFIED:
                continue

            # Find neighbors for the current point
            p1 = coords[i]
            neighbors_indices = []
            for j, p2 in enumerate(coords):
                if i == j:
                    continue
                # Using squared Euclidean distance to avoid sqrt for performance
                if (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 <= eps**2:
                    neighbors_indices.append(j)

            # If not enough neighbors, mark as noise (for now)
            if len(neighbors_indices) < min_samples - 1:
                labels[i] = NOISE
                continue

            # This is a core point, start a new cluster
            cluster_id += 1
            labels[i] = cluster_id
            
            # Expand the cluster from the seed set of neighbors
            seed_set = set(neighbors_indices)
            while seed_set:
                q_idx = seed_set.pop()

                # If the neighbor was previously marked as noise, it's a border point
                if labels[q_idx] == NOISE:
                    labels[q_idx] = cluster_id
                
                # If already classified, skip
                if labels[q_idx] != UNCLASSIFIED:
                    continue

                # Add the point to the current cluster
                labels[q_idx] = cluster_id
                
                # Find its neighbors to potentially expand the cluster further
                q_p1 = coords[q_idx]
                q_neighbors_indices = []
                for j, q_p2 in enumerate(coords):
                    if (q_p1[0] - q_p2[0])**2 + (q_p1[1] - q_p2[1])**2 <= eps**2:
                        q_neighbors_indices.append(j)

                # If this neighbor is also a core point, add its neighbors to the seed set
                if len(q_neighbors_indices) >= min_samples - 1:
                    seed_set.update(q_neighbors_indices)
        
        return labels
    def extract_caption(self, input_text: str, left_index: int = None, right_index: int = None) -> dict[str, str]:
        """
        Processes a JSON string of text recognized in video frames to extract captions.

        Args:
            input_text: A JSON string containing frame-by-frame text data.
            left_index: The starting frame index for a "middle" section.
            right_index: The ending frame index for a "middle" section.

        Returns:
            A dictionary containing the extracted captions, potentially split into
            "left", "middle", and "right" sections if indices are provided.
        """
        try:
            data = json.loads(input_text)

            # Flatten the data and apply cleaning/filtering
            all_texts = []
            for item in data:
                for text_info in item.get('texts', []):
                    original_text = text_info.get('text')
                    if not original_text:
                        continue
                    
                    # Clean the text using the helper method
                    cleaned_text = self._clean_text(original_text)

                    # Only proceed if the cleaned text is not empty
                    if cleaned_text.strip():
                        loc = text_info.get('location', {})
                        all_texts.append({
                            'text': cleaned_text, 
                            'width': loc.get('widthInPixel'),
                            'height': loc.get('heightInPixel'),
                            'top': loc.get('topOffsetInPixel'),
                            'left': loc.get('leftOffsetInPixel'),
                            'frame_index': item.get('index')
                        })
            
            if not all_texts:
                return {"result": "[]"}

            # Calculate center coordinates and filter for the lower region of the frame
            for text_info in all_texts:
                text_info['center_x'] = text_info['left'] + text_info['width'] / 2
                text_info['center_y'] = text_info['top'] + text_info['height'] / 2
            
            max_y = max(t['top'] + t['height'] for t in all_texts)
            lower_region_texts = [t for t in all_texts if t['center_y'] > max_y * 0.4]

            if not lower_region_texts:
                return {"result": "[]"}

            # Perform DBSCAN clustering
            coords = [[t['center_x'], t['center_y']] for t in lower_region_texts]
            cluster_labels = self._custom_dbscan(coords, eps=50, min_samples=2)
            
            for i, text_info in enumerate(lower_region_texts):
                text_info['cluster'] = cluster_labels[i]

            # Filter out noise and find the largest cluster (most likely the captions)
            clustered_texts = [t for t in lower_region_texts if t['cluster'] != -1]
            if not clustered_texts:
                return {"result": "[]"}

            cluster_counts = Counter(t['cluster'] for t in clustered_texts)
            if not cluster_counts:
                 return {"result": "[]"}
            largest_cluster_label = cluster_counts.most_common(1)[0][0]
            captions = [t for t in clustered_texts if t['cluster'] == largest_cluster_label]

            captions.sort(key=lambda x: x['frame_index'])
            
            if not captions:
                return {"result": "[]"}
            print("Before dedup:", [(c['frame_index'], c['text']) for c in captions])
            deduplicated_captions = [captions[0]]
            
            # Max gap between frames for captions to be considered consecutive
            # FRAME_GAP_TOLERANCE = 2

            for i in range(1, len(captions)):
                current_cap = captions[i]
                last_kept_cap = deduplicated_captions[-1]

                # Check if it's a duplicate of the last kept caption
                is_same_text = current_cap['text'] == last_kept_cap['text']
                # is_consecutive_frame = (current_cap['frame_index'] - last_kept_cap['frame_index']) <= FRAME_GAP_TOLERANCE

                # If the text is the same and the frame is consecutive, skip it
                if is_same_text: # and is_consecutive_frame
                    continue
                
                # Otherwise, it's a new, unique caption, so we keep it
                deduplicated_captions.append(current_cap)

            print("After dedup:", [(c['frame_index'], c['text']) for c in deduplicated_captions])
            # Group by frame index and format the final output
            output_by_frame = defaultdict(list)
            for cap in deduplicated_captions:
                output_by_frame[cap['frame_index']].append(cap)
            
            output_data = []
            for frame_idx in sorted(output_by_frame.keys()):
                texts_list = []
                for row in output_by_frame[frame_idx]:
                    texts_list.append({
                        "text": row['text'],
                        "location": {
                            "widthInPixel": int(row['width']),
                            "heightInPixel": int(row['height']),
                            "topOffsetInPixel": int(row['top']),
                            "leftOffsetInPixel": int(row['left'])
                        }
                    })
                output_data.append({
                    "index": int(frame_idx),
                    "texts": texts_list
                })

            # Split the output if left/right indices are provided
            if left_index is not None and right_index is not None:
                left = [item for item in output_data if item['index'] < left_index]
                middle = [item for item in output_data if left_index <= item['index'] < right_index]
                right = [item for item in output_data if item['index'] >= right_index]

                return {
                    "left": json.dumps(left, ensure_ascii=False, indent=2),
                    "middle": json.dumps(middle, ensure_ascii=False, indent=2),
                    "right": json.dumps(right, ensure_ascii=False, indent=2)
                }
            
            result = json.dumps(output_data, ensure_ascii=False, indent=2)
            return {"result": result}
            
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            raise ValueError(f"Caption extraction failed: {str(e)}\nTraceback:\n{tb_str}")


    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:

        raw_ocr_text = tool_parameters.get("raw_ocr_text")
        left_index = tool_parameters.get("left_index")
        right_index = tool_parameters.get("right_index")
        # print(f"in invoke: left: {left_index}, right: {right_index}") 
        if not raw_ocr_text:
            yield self.create_json_message({
                "result": "请提供原始的OCR结果"
            })
            return
        if left_index is not None or right_index is not None:
            if left_index is None or right_index is None:
                yield self.create_json_message({
                    "result": "请同时提供左入点和右出点的索引"
                })
                return
            result = self.extract_caption(raw_ocr_text, left_index, right_index)
            yield self.create_json_message(result)
        
        result = self.extract_caption(raw_ocr_text, left_index, right_index)
        yield self.create_json_message(result)
