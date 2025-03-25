from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
import logging

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
)

# 实体类型到颜色的映射 V1
# en_color = {
#     'abstract': {"background": '#5A98D0', "border": '#1B3B6F'},  # 赛博蓝/深海蓝
#     'chemical': {"background": '#35A29F', "border": '#1F6F78'},  # 青绿/深青蓝
#     'disease': {"background": '#FF3864', "border": '#A30052'},  # 霓虹红/暗紫红 (主色)
#     'gene': {"background": '#FFCE4F', "border": '#D48C00'},  # 明亮黄/金黄
#     'metabolite': {"background": '#E457A6', "border": '#9C1B6C'},  # 霓虹粉/暗粉紫
#     'pathway': {"background": '#8A5CF6', "border": '#4E2A84'},  # 科技紫/深紫
#     'processes': {"background": '#6BE3C3', "border": '#2A9D8F'},  # 赛博绿/深绿
#     'protein': {"background": '#1E88E5', "border": '#0D47A1'},  # 电蓝/深蓝
#     'region': {"background": '#FDCB58', "border": '#C67C00'},  # 柔和金色/深金黄
# }

# 实体类型到颜色的映射 V2
en_color = {
    'abstract': {"background": '#59A14E', "border": '#59A14E'},  
    'chemical': {"background": '#4E79A7', "border": '#4E79A7'}, 
    'disease': {"background": '#E15759', "border": '#E15759'},  
    'gene': {"background": '#EDC949', "border": '#EDC949'},  
    'metabolite': {"background": '#AF7AA1', "border": '#AF7AA1'},  
    'pathway': {"background": '#76B7B2', "border": '#76B7B2'}, 
    'processes': {"background": '#FF9DA7', "border": '#FF9DA7'},  
    'protein': {"background": '#F28E2C', "border": '#F28E2C'},  
    'region': {"background": '#9E7A65', "border": '#9E7A65'}, 
}

cont = {
'disease': 196769,
 'gene': 8802,
 'chemical': 39106,
 'abstract': 46835,
 'metabolite': 156,
 'protein': 435,
 'processes': 225
 }

def scale_entity_size_log(entity_relation_counts):
    for entity, count in entity_relation_counts.items():
        entity_relation_counts[entity] = (count-1)*8 + 25
    return entity_relation_counts


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


def kg_visualization(pmid_list: List, kg_dict: Dict[str, Dict]) -> Dict:
    """
    从 kg_dict 构建用于可视化的知识图谱数据。

    Args:
        kg_dict: 包含知识图谱数据的字典，键是 PMID，值是包含 'entities' 和 'relations' 键的字典。
        en_color: 实体类型到颜色的映射。

    Returns:
        包含 'nodes' 和 'edges' 键的字典，用于可视化。
    """
    entities_dict = defaultdict(lambda: {"id": len(entities_dict) + 1})  # 自动生成 ID
    relations_set: Set[Tuple[str, str, str]] = set()  # 使用集合存储关系，避免重复
    relations_list: List[Dict] = []

    entity_relation_counts = {}
    for pmid in pmid_list:
        for en in kg_dict[pmid]['entities']:
            # 如果实体名称不在 entities_dict 中, 则创建新的 Entity 对象
            if en.name not in entities_dict:
                entity = Entity(name=en.name, label=en.label, color=en_color.get(en.label, {}).get("background", "gray"))
                en_title = en.properties_info
                en_title['name']= en.name
                en_title['label']= en.label
                if en.label == "abstract":
                    en_title = {"name": en.name, "label": en.label}

                entities_dict[en.name] = {
                    "id": entities_dict[en.name]["id"], 
                    "label": entity.name, 
                    "color": entity.color,
                    "title": format_dict_to_html_br(en_title)
                    }
                
        
        for rel in kg_dict[pmid]['relations']:
            relation_tuple = (rel.startEntity.name, rel.endEntity.name, rel.name)
            if relation_tuple not in relations_set:
                relations_set.add(relation_tuple)
                re_title = rel.properties_info
                re_title['edges'] = rel.name
                relations_list.append({
                    "from": entities_dict[rel.startEntity.name]["id"],
                    "to": entities_dict[rel.endEntity.name]["id"],
                    "label": rel.name,
                    "title": format_dict_to_html_br(re_title)
                })
            
            # 计算每个实体连接的关系的数量
            entity_relation_counts[rel.startEntity.name] = entity_relation_counts.get(rel.startEntity.name,0) + 1
            entity_relation_counts[rel.endEntity.name] = entity_relation_counts.get(rel.endEntity.name, 0) + 1
        
    logging.info('entity_relation_counts', entity_relation_counts)
    entity_relation_counts = scale_entity_size_log(entity_relation_counts)
    logging.info('entity_relation_counts', entity_relation_counts)
    for en_name, _ in entities_dict.items():
        if en_name in entity_relation_counts:
            entities_dict[en_name]["size"] = entity_relation_counts[en_name]
        else:
            entities_dict[en_name]["size"] = 25
                
    output_kg = {
        'nodes': list(entities_dict.values()),
        'edges': relations_list
    }
    return output_kg