from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any
import logging
import math

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
)

# 实体类型到颜色的映射 V1
en_color = {
    'article': {"background": '#A7CBFC', "border": '#1B3B6F'},  # 赛博蓝/深海蓝
    'chemical': {"background": '#35A29F', "border": '#1F6F78'},  # 青绿/深青蓝
    'disease': {"background": '#FF3864', "border": '#A30052'},  # 霓虹红/暗紫红 (主色)
    'gene': {"background": '#FFCE4F', "border": '#D48C00'},  # 明亮黄/金黄
    'metabolite': {"background": '#E457A6', "border": '#9C1B6C'},  # 霓虹粉/暗粉紫
    'pathway': {"background": '#8A5CF6', "border": '#4E2A84'},  # 科技紫/深紫
    'processes': {"background": '#6BE3C3', "border": '#2A9D8F'},  # 赛博绿/深绿
    'protein': {"background": '#1E88E5', "border": '#0D47A1'},  # 电蓝/深蓝
    'region': {"background": '#9E7A65', "border": '#9E7A65'},  
}

# 实体类型到颜色的映射 V2
# en_color = {
#     'abstract': {"background": '#59A14E', "border": '#59A14E'},  
#     'chemical': {"background": '#4E79A7', "border": '#4E79A7'}, 
#     'disease': {"background": '#E15759', "border": '#E15759'},  
#     'gene': {"background": '#EDC949', "border": '#EDC949'},  
#     'metabolite': {"background": '#AF7AA1', "border": '#AF7AA1'},  
#     'pathway': {"background": '#76B7B2', "border": '#76B7B2'}, 
#     'processes': {"background": '#FF9DA7', "border": '#FF9DA7'},  
#     'protein': {"background": '#F28E2C', "border": '#F28E2C'},  
#     'region': {"background": '#9E7A65', "border": '#9E7A65'}, 
# }

cont = {
    'disease': 7895,
    'gene': 19549,
    'article': 153060,
    'chemical': 10295,
    'protein': 7062,
    'region': 3419,
    'processes': 5170,
    'pathway': 3128,
    'metabolite': 2170
    }

def scale_entity_size_log(entity_relation_counts):
    max_size = 100
    min_size=45
    
    scaled_counts = {}
    if not entity_relation_counts:
        return scaled_counts # Return empty dict if input is empty

    # Determine min_count and max_count from the actual data
    all_counts = list(entity_relation_counts.values())
    if not all_counts: # Should not happen if entity_relation_counts wasn't empty, but safe check
        return scaled_counts

    actual_min_count = min(all_counts)
    actual_max_count = max(all_counts)

    output_range = float(max_size - min_size)
    input_range = float(actual_max_count - actual_min_count)

    # Handle case where all counts are the same
    if input_range == 0:
        middle_size = int(round((min_size + max_size) / 2.0))
        for entity in entity_relation_counts:
            scaled_counts[entity] = middle_size
        return scaled_counts

    # Perform linear scaling for each entity
    for entity, count in entity_relation_counts.items():
        # No clamping needed here as count is guaranteed to be within [actual_min_count, actual_max_count]
        fraction = (float(count) - actual_min_count) / input_range
        scaled_size_float = min_size + fraction * output_range
        scaled_counts[entity] = int(round(scaled_size_float))

    return scaled_counts



def scale_edge_length(head_entity, tail_entity, entity_relation_counts):
    """
    根据相连实体的计数值（count）计算边的缩放后长度。

    长度在 [min_length, max_length] 范围内进行线性缩放。
    缩放基于两个连接实体（头实体或尾实体）中 *较大* 的计数值。
    用于缩放的输入范围由 entity_relation_counts 中 *所有* 实体的最小和最大计数值确定。

    Args:
        head_entity (str): 头实体的标识符。
        tail_entity (str): 尾实体的标识符。
        entity_relation_counts (dict): 一个字典，键是实体标识符，值是它们关联的计数值
        （例如，关联的关系数量）。

    Returns:
        int: 缩放后的边长度，四舍五入到最近的整数，范围在 [min_length, max_length] 内。
            如果无法确定计数值或所有计数值都相同，则返回中间值。
    """
    min_length = 100  # 边的最小长度
    max_length = 1000  # 边的最大长度

    # --- 基本检查 ---
    if not entity_relation_counts:
        # 无法确定计数或范围，返回默认的中间长度
        logging.info("警告: entity_relation_counts 为空。返回默认长度。")
        return min_length
    all_counts = list(entity_relation_counts.values())
    if not all_counts:
        # 如果 entity_relation_counts 不为空但没有值（理论上不太可能），进行安全检查
        logging.info("警告: entity_relation_counts 有键但没有值？返回默认长度。")
        return min_length


    actual_min_count = min(all_counts)
    actual_max_count = max(all_counts)

    head_count = entity_relation_counts.get(head_entity, 0)
    tail_count = entity_relation_counts.get(tail_entity, 0)

    # 边的长度取决于具有 *较大* 计数的实体
    relevant_count = max(head_count, tail_count)

    min_weight=0.7  # 头实体的默认权重
    max_weight=0.3
    total_weight = min_weight + max_weight
    
    if math.isclose(total_weight, 0.0): # 检查权重和是否为零
        # Handle zero weight sum case
        return min_length
    
    if head_count > tail_count:
        relevant_count = (head_count * max_weight + tail_count * min_weight) / total_weight
    else:
        relevant_count = (head_count * min_weight + tail_count * max_weight) / total_weight

    # --- 处理缩放 ---
    output_range = float(max_length - min_length) 
    input_range = float(actual_max_count - actual_min_count) # 输入范围

    # 处理所有原始实体计数值都相同的情况
    if input_range == 0:
        return min_length

    # --- 执行线性缩放 ---
    clamped_count = max(actual_min_count, min(relevant_count, actual_max_count))

    # 计算输入范围内的比例
    fraction = (float(clamped_count) - actual_min_count) / input_range

    # 再次确保 fraction 在 [0.0, 1.0] 范围内 (防御性编程)
    fraction = max(0.0, min(fraction, 1.0))

    # 计算缩放后的长度（浮点数）
    scaled_length_float = min_length + fraction * output_range

    # 返回四舍五入后的整数值
    return int(round(scaled_length_float))


def format_dict_to_html_br(input_dict: Dict) -> str:
    """
    将字典转换为带有 HTML 换行符 (<br>) 的字符串。

    Args:
        input_dict: 输入的字典。

    Returns:
        格式化后的字符串。
    """
    lines = []
    for key, value in input_dict.items():
        # 使用 HTML 实体 ' 替代单引号，以避免在 HTML 中出现问题
        # 同时处理 value 中可能出现的单引号
        safe_key = str(key).replace("'", "'")
        safe_value = str(value).replace("'", "'")
        lines.append(f"{safe_key}: {safe_value}")

    return "<br>".join(lines) + "<br>"

@dataclass
class Entity:
    name: str
    label: str
    id: int = field(default=None, repr=False)  # ID 在创建后设置, 不参与 repr
    color: str = field(default=None, repr=False)

@dataclass
class Relation:
    start_entity: Entity
    end_entity: Entity
    name: str


# def kg_visualization(pmid_list: List, kg_dict: Dict[str, Dict]) -> Dict:
#     """
#     从 kg_dict 构建用于可视化的知识图谱数据。

#     Args:
#         kg_dict: 包含知识图谱数据的字典，键是 PMID，值是包含 'entities' 和 'relations' 键的字典。
#         en_color: 实体类型到颜色的映射。

#     Returns:
#         包含 'nodes' 和 'edges' 键的字典，用于可视化。
#     """
#     entities_dict = defaultdict(lambda: {"id": len(entities_dict) + 1})  # 自动生成 ID
#     relations_set: Set[Tuple[str, str, str]] = set()  # 使用集合存储关系，避免重复
#     relations_list: List[Dict] = []

#     entity_relation_counts = {}
    
#     related_pairs = set()
#     entity_name_mapping = {}
#     for pmid in pmid_list:
#         for en in kg_dict[pmid]['entities']:
            
                
                
#             if en.name not in entities_dict:
                
#                 entity = Entity(name=en.name, label=en.label, color=en_color.get(en.label, {}).get("background", "gray"))
#                 en_title = en.properties_info
#                 en_title['name']= en.name
#                 en_title['label']= en.label
#                 if en.label == "abstract":
#                     en_title = {"name": en.name, "label": "article"}

#                     entities_dict[en.name] = {
#                         "id": entities_dict[en.name]["id"], 
#                         "label": entity.name, 
#                         "color": entity.color,
#                         "title": format_dict_to_html_br(en_title),
#                         "group": "article"
#                         }
#                 else:
#                     entities_dict[en.name] = {
#                         "id": entities_dict[en.name]["id"], 
#                         "label": entity.name, 
#                         "color": entity.color,
#                         "title": format_dict_to_html_br(en_title),
#                         "group": en.label
#                         }
                
        
#         for rel in kg_dict[pmid]['relations']:
#             relation_tuple = (rel.startEntity.name, rel.endEntity.name, rel.name)
#             if relation_tuple not in relations_set:
#                 relations_set.add(relation_tuple)
#                 re_title = rel.properties_info
#                 re_title['edges'] = rel.name
#                 relations_list.append({
#                     "from": rel.startEntity.name,
#                     "to": rel.endEntity.name,
#                     "label": rel.name,
#                     "title": format_dict_to_html_br(re_title),
#                     "length": 300 
#                 })
            
#             # 计算每个实体连接的关系的数量
#             related_pairs1 = f"{rel.startEntity.name}_{rel.endEntity.name}"
#             related_pairs2 = f"{rel.endEntity.name}_{rel.startEntity.name}"
            
#             if related_pairs1 not in related_pairs and related_pairs2 not in related_pairs:
#                 related_pairs.add(related_pairs1)
#                 related_pairs.add(related_pairs2)
#                 entity_relation_counts[rel.startEntity.name] = entity_relation_counts.get(rel.startEntity.name, 0) + 1
#                 entity_relation_counts[rel.endEntity.name] = entity_relation_counts.get(rel.endEntity.name, 0) + 1
    
#     entity_relation_counts = scale_entity_size_log(entity_relation_counts)
#     for en_name, _ in entities_dict.items():
#         if en_name in entity_relation_counts:
#             entities_dict[en_name]["size"] = entity_relation_counts[en_name]
#         else:
#             entities_dict[en_name]["size"] = 25
    
#     # 计算边的长度
#     relations_list_length = []

#     for re in relations_list:
#         head_entity = re["from"]
#         tail_entity = re["to"]
#         re["length"] = scale_edge_length(
#             head_entity, 
#             tail_entity, 
#             entity_relation_counts
#             )
#         re["from"] = entities_dict[head_entity]["id"]
#         re["to"] = entities_dict[tail_entity]["id"]
#         relations_list_length.append(re)
    
#     output_kg = {
#         'nodes': list(entities_dict.values()),
#         'edges': relations_list_length
#     }
#     return output_kg


def kg_visualization(pmid_list: List[str], kg_dict: Dict[str, Dict[str, List[Any]]]) -> Dict:
    """
    从 kg_dict 构建用于可视化的知识图谱数据, 并根据 unique_id 统一实体名称。

    Args:
        pmid_list: 要包含的 PMID 列表。
        kg_dict: 包含知识图谱数据的字典，键是 PMID，值是包含 'entities' 和 'relations' 键的字典。
                'entities' 是 Entity 对象列表, 'relations' 是 Relation 对象列表。

    Returns:
        包含 'nodes' 和 'edges' 键的字典，用于可视化。
    """
    # --- Step 1: Map original names to canonical names based on unique_id ---
    entity_name_mapping: Dict[str, str] = {} # original_name -> canonical_name
    unique_id_to_canonical_name: Dict[str, str] = {} # unique_id -> first_seen_name (canonical)

    for pmid in pmid_list:
        if pmid not in kg_dict: continue # Skip if pmid not in kg_dict
        for en in kg_dict[pmid].get('entities', []):
            original_name = en.name
            unique_id = en.properties_info.get('unique_id')

            if unique_id:
                if unique_id in unique_id_to_canonical_name:
                    # This unique_id has been seen, use its established canonical name
                    canonical_name = unique_id_to_canonical_name[unique_id]
                else:
                    # First time seeing this unique_id, establish its canonical name
                    canonical_name = original_name
                    unique_id_to_canonical_name[unique_id] = canonical_name
                # Map the original name to the canonical name
                entity_name_mapping[original_name] = canonical_name
            else:
                # No unique_id, the name is its own canonical name (unless already mapped by unique_id elsewhere)
                if original_name not in entity_name_mapping:
                    entity_name_mapping[original_name] = original_name


    # --- Step 2: Build nodes and edges using canonical names ---
    # Use canonical names as keys for entities_dict
    entities_dict = defaultdict(lambda: {"id": len(entities_dict) + 1})
    relations_set: Set[Tuple[str, str, str]] = set()  # Uses canonical names
    relations_list: List[Dict] = [] # Stores relations with canonical names initially

    entity_relation_counts = defaultdict(int) # Counts based on canonical names
    related_pairs = set() # Tracks pairs based on canonical names

    for pmid in pmid_list:
        if pmid not in kg_dict: continue

        # Process entities to populate entities_dict using canonical names
        for en in kg_dict[pmid].get('entities', []):
            original_name = en.name
            # Get the canonical name using the mapping, default to original if somehow not mapped
            canonical_name = entity_name_mapping.get(original_name, original_name)

            if canonical_name not in entities_dict:
                # Fetch properties from the original entity object 'en'
                entity_label_type = en.label
                entity_color = en_color.get(entity_label_type, {}).get("background", "gray")
                en_title_props = en.properties_info.copy() # Important: copy properties
                #en_title_props['original_name'] = original_name # Optionally add original name
                en_title_props['name'] = canonical_name # Add canonical name
                en_title_props['label'] = entity_label_type

                node_group = entity_label_type
                node_label = canonical_name # Use canonical name for the node label

                if entity_label_type == "abstract":
                    # Keep specific handling for abstract if needed
                    en_title = {"name": canonical_name, "label": "article"}
                    node_group = "article"
                    node_label = canonical_name # Ensure label is canonical here too

                else:
                    # Prepare title dictionary using potentially modified props
                    en_title = en_title_props
                    
                entities_dict[canonical_name] = {
                    "id": entities_dict[canonical_name]["id"], # Get the auto-generated ID
                    "label": node_label,
                    "color": entity_color,
                    "title": format_dict_to_html_br(en_title),
                    "group": node_group
                }

        # Process relations using canonical names
        for rel in kg_dict[pmid].get('relations', []):
            original_start_name = rel.startEntity.name
            original_end_name = rel.endEntity.name

            # Get canonical names for relation endpoints
            canonical_start_name = entity_name_mapping.get(original_start_name, original_start_name)
            canonical_end_name = entity_name_mapping.get(original_end_name, original_end_name)

            # Use canonical names for relation tuple and processing
            relation_tuple = (canonical_start_name, canonical_end_name, rel.name)

            if relation_tuple not in relations_set:
                relations_set.add(relation_tuple)
                re_title = rel.properties_info.copy() # Copy properties
                re_title['relationship'] = rel.name # Use a different key than 'edges' maybe?


                relations_list.append({
                    "from": canonical_start_name, # Store canonical name temporarily
                    "to": canonical_end_name,   # Store canonical name temporarily
                    "label": rel.name,
                    "title": format_dict_to_html_br(re_title),
                    # "length": 300 # Length will be calculated later
                })

            # Calculate relation counts based on canonical names
            related_pair_key1 = f"{canonical_start_name}_{canonical_end_name}"
            related_pair_key2 = f"{canonical_end_name}_{canonical_start_name}"

            # Count relations per canonical entity only once per pair
            if related_pair_key1 not in related_pairs and related_pair_key2 not in related_pairs:
                related_pairs.add(related_pair_key1)
                # Increment counts for the canonical names
                entity_relation_counts[canonical_start_name] += 1
                entity_relation_counts[canonical_end_name] += 1


    # --- Step 3: Calculate Node Sizes based on canonical entity counts ---
    # Note: scale_entity_size_log receives counts keyed by canonical names
    scaled_entity_sizes = scale_entity_size_log(entity_relation_counts)

    for canonical_name, node_data in entities_dict.items():
        # Use canonical name to look up size
        node_data["size"] = scaled_entity_sizes.get(canonical_name, 25) # Default size if no relations


    # --- Step 4: Calculate Edge Lengths and Finalize Output ---
    final_edges = []
    for rel_data in relations_list:
        # Retrieve canonical names stored temporarily
        head_canonical_name = rel_data["from"]
        tail_canonical_name = rel_data["to"]

        # Calculate length using canonical names and scaled sizes (implicitly via counts)
        # Note: scale_edge_length receives canonical names and counts keyed by canonical names
        rel_data["length"] = scale_edge_length(
            head_canonical_name,
            tail_canonical_name,
            entity_relation_counts # Pass the original counts dictionary
        )

        # Replace canonical names with their corresponding node IDs
        # Lookup in entities_dict uses canonical names
        rel_data["from"] = entities_dict[head_canonical_name]["id"]
        rel_data["to"] = entities_dict[tail_canonical_name]["id"]

        final_edges.append(rel_data)

    output_kg = {
        'nodes': list(entities_dict.values()),
        'edges': final_edges
    }
    return output_kg
