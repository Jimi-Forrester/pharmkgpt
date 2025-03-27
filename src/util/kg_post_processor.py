import networkx as nx
import logging
from typing import List,Dict,Optional,Set
from FlagEmbedding import FlagReranker
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher
dispatcher = get_dispatcher(__name__)


# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
)

from builtins import print as _print
from sys import _getframe
def print(*arg, **kw):
    s = f'Line {_getframe(1).f_lineno}'
    return _print(f"Func {__name__} - {s}", *arg, **kw)

import string
def ngram_overlap(span,sent,n=3):
    while (len(span)<n) or (len(sent)<n):
        n -= 1
    if n<=0:
        return 0.0
    span = span.lower()
    sent = sent.lower()
    span_tokens = [token for token in span.split() if token not in string.punctuation]
    span_tokens = ''.join(span_tokens)
    sent_tokens = [token for token in sent.split() if token not in string.punctuation]
    sent_tokens = ''.join(sent_tokens)
    span_tokens = set([span_tokens[i:i+n] for i in range(len(span_tokens)-n+1)])
    sent_tokens = set([sent_tokens[i:i+n] for i in range(len(sent_tokens)-n+1)])
    overlap = span_tokens.intersection(sent_tokens)
    return float((len(overlap)+0.01)/(len(span_tokens)+0.01))

class NaivePostprocessor(BaseNodePostprocessor):
    """Naive Node Postprocessor."""

    dataset: str = Field

    @classmethod
    def class_name(cls) -> str:
        return "NaivePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """对检索到的节点（nodes）进行一个简单的后处理"""
        
        logging.info('==============NaivePostprocessor==============')
        entity_order = {}
        sorted_nodes = []
        
        # 构建排序元组
        for i, node in enumerate(nodes):
            node_id = node.node.id_
            ent  = node_id
            ctx_seq = i
            if ent not in entity_order:
                entity_order[ent] = len(entity_order)
            sorted_nodes.append((ent,ctx_seq,node))
        
        # logging.info(f"entity_order: {entity_order}")
        # 按照 abstarct 的顺序对节点内容进行排序
        sorted_nodes.sort(key=lambda x:(entity_order[x[0]],x[1]))
        sorted_nodes = [node for _,_,node in sorted_nodes]
        logging.info(f"sorted_nodes: {len(sorted_nodes)}")
        
        prev_ent = ''
        for i in range(0,len(sorted_nodes)):
            temp_ent = sorted_nodes[i].node.id_
            if (prev_ent == temp_ent):
                sorted_nodes[i].node.text = sorted_nodes[i].node.text[len(temp_ent+': '):]
            if i<len(sorted_nodes)-1:
                next_ent = sorted_nodes[i+1].node.id_
                if next_ent!=temp_ent:
                    sorted_nodes[i].node.text += '\n'
            prev_ent = temp_ent

        logging.info(f"===NaivePostprocessor output===: {len(sorted_nodes)}")
        return sorted_nodes
    
class KGRetrievePostProcessor(BaseNodePostprocessor):
    """
    继承BaseNodePostprocessor
    直接找到与query相关的实体,然后找到与这些实体相关的文本
    """

    dataset: str = Field
    ents: Set[str] = Field
    doc2kg: Dict[str,list] = Field
    chunks_index: Dict[str,str] = Field
    hops: int = Field

    @classmethod
    def class_name(cls) -> str:
        return "KGRetrievePostprocessor"
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        logging.info('==============KGRetrievePostProcessor==============')
        top_k = len(nodes)
        
        # 检索到的实体
        retrieved_ids = set()
        retrieved_ents = set()

        # 相关的实体
        related_ents = set()
        
        # 最相关的实体
        highly_related_ents = set()

        textid2score = dict()
        ent_count = dict()
        ent_score = dict()

        # ----------------------找到与query相关的实体----------------------
        for i in range(len(nodes)):
            node = nodes[i]
            node_id = node.node.id_
            
            retrieved_ids.add(node_id)
            
            textid2score[node_id] = node.score
            entity = node.node.id_

            # 排名靠前一般的实体认为是最相关的
            if (i<(top_k//2)) and (entity in retrieved_ents):
                highly_related_ents.add(entity)
            retrieved_ents.add(entity)
            
            # 记录最相关实体出现的次数和得分
            if entity not in ent_count:
                ent_count[entity] = 0
                ent_score[entity] = 0.0
            ent_count[entity] += 1
            ent_score[entity] += node.score

        
        sorted_ents = sorted(ent_count.keys(),key=lambda x:(ent_score[x]/ent_count[x]),reverse=True)
        
        for i in range(min(2,len(sorted_ents))):
            highly_related_ents.add(sorted_ents[i])

        logging.info(f'highly_related_ents:{highly_related_ents}')
        logging.info(f'retrieved_ents: {retrieved_ents}')

        logging.info("根据相似的实体，和实体关系，找到相关的实体。")
        additional_ents = set()
        for node in nodes:
            entity = node.node.id_
            # logging.info(f'####entity:{ entity}')
            # logging.info(entity not in self.doc2kg)
            if (entity not in self.doc2kg) or (len(self.doc2kg[entity])==0):
                continue
            for triplet in self.doc2kg[entity]:
                # logging.info(f'####triplet: {triplet}')
                h,r,t = triplet
                triplet = [h,r,t]
                if (h in self.ents) and (h not in retrieved_ents):
                    additional_ents.add(h)
                    if h not in ent_count:
                        ent_count[h] = 0
                        ent_score[h] = 0.0
                    ent_count[h] += 1
                    ent_score[h] += node.score
                if (t in self.ents) and (t not in retrieved_ents):
                    additional_ents.add(t)
                    if t not in ent_count:
                        ent_count[t] = 0
                        ent_score[t] = 0.0
                    ent_count[t] += 1
                    ent_score[t] += node.score

        additional_ents = additional_ents.union(retrieved_ents)
        logging.info(f'additional_ents: {additional_ents}')
        
        # ----------------------多跳扩展----------------------
        
        logging.info("Start multi-hop expansion.")
        for hop in range(self.hops):
            logging.info(f"!!hop!!!: {hop}")
            # 基于初始相关实体更新相关实体
            related_ents = related_ents.union(additional_ents)

            # 初始相关实体
            temp_ents = set(additional_ents)
            additional_ents = set()
            for ent in temp_ents:
                    if ent in retrieved_ids:
                        continue
                    
                    for triplet in self.doc2kg[ent]:
                        h,r,t = triplet
                        if (h in self.ents) and (h not in related_ents):
                            additional_ents.add(h)
                            if h not in ent_count:
                                ent_count[h] = 0
                                ent_score[h] = 0.0
                            ent_count[h] += 1
                            ent_score[h] += float((ent_score[ent])/(ent_count[ent]))
                        if (t in self.ents) and (t not in related_ents):
                            additional_ents.add(t)
                            if t not in ent_count:
                                ent_count[t] = 0
                                ent_score[t] = 0.0
                            ent_count[t] += 1
                            ent_score[t] += float((ent_score[ent])/(ent_count[ent]))

        related_ents = related_ents.union(additional_ents)

        logging.info(f"multi-hop expansion: {related_ents}")
        
        additional_ids = set()
        avg_score = float(sum([node.score for node in nodes])/len(nodes))
        retrieved_ents = retrieved_ents-highly_related_ents

        logging.info(f"increase ent : {related_ents-retrieved_ents}")
        # 根据关系找到对应 PMID
        for ent in (related_ents-retrieved_ents):
            if (ent not in self.chunks_index) or (len(self.chunks_index[ent])==0) or (ent not in self.doc2kg):
                continue
            ctx_id = ent
            if ctx_id in retrieved_ids:
                continue
            additional_ids.add(ctx_id)
            textid2score[ctx_id] = 0.0
            if ent in ent_score:
                textid2score[ctx_id] += (ent_score[ent]+avg_score)/(ent_count[ent]+1)

        logging.info(f"textid2score: {textid2score}")
        logging.info(f'additional_ids: {additional_ids}')
        
        added_nodes = []
        for pmid_indx in additional_ids:
            if pmid_indx in self.chunks_index:
                ctx_text = self.chunks_index[pmid_indx]
                node = TextNode(id_=ctx_id,text=ctx_text)
                node = NodeWithScore(node=node,score=textid2score[ctx_id])
                added_nodes.append(node)
                
        added_nodes = sorted(added_nodes,key=lambda x:x.score,reverse=True)
        nodes = nodes+added_nodes
        
        logging.info(f"===KGRetrieve output===: {len(nodes)}")
        return nodes

class GraphFilterPostProcessor(BaseNodePostprocessor):

    """KnowledgeGraph-based Node processor."""
    dataset: str = Field
    topk: int = Field
    use_tpt: bool = Field
    ents: Set[str] = Field
    doc2kg: Dict[str,list] = Field
    chunks_index: Dict[str,str] = Field
    reranker: FlagReranker = Field

    @classmethod
    def class_name(cls) -> str:
        return "GraphFilterPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """基于知识图谱（KG）对初步检索得到的节点（nodes）进行过滤和重排序"""
        
        # 实体
        logging.info('==============GraphFilterPostProcessor==============')
        ents = set()
        
        # 关系
        rels = set()

        g = nx.MultiGraph()

        for node in nodes:
            if len(node.node.id_.split('##')) == 2:
                ent, pmid_indx = node.node.id_.split('##')
            else:
                ent  = node.node.id_
                pmid_indx = ent
                
            ents.add(ent)
        
        logging.info(f"根据初始节点得到的实体")

        # 构建了一个小的知识图谱
        for node in nodes:
            if len(node.node.id_.split('##')) == 2:
                ent, pmid = node.node.id_.split('##')
            else:
                ent  = node.node.id_
                pmid = ent
                
            ents.add(ent)
            
            # 好像没有考虑到多个关系的情况
            if (ent not in self.doc2kg) or (pmid not in self.doc2kg) or (len(self.doc2kg[ent])==0):
                continue
            
            for triplet in self.doc2kg[ent]:
                h,r,t = triplet
                h = h.strip()
                r = r.strip()
                t = t.strip()
                triplet = [h,r,t]
                ents.add(h)
                ents.add(t)
                rels.add(r)
                g.add_edge(h,t,rel=r,source=node.node.id_,weight=node.score)
        

        mentioned_ents = set()
        mentioned_rels = set()

        # 如果实体/关系与查询字符串的 n-gram 重叠度大于或等于 0.90，则认为该实体/关系在查询中被 明确提及。
        for ent in ents:
            overlap_score = ngram_overlap(ent,query_bundle.query_str)
            if overlap_score>=0.90:
                mentioned_ents.add(ent)
        
        for rel in rels:
            overlap_score = ngram_overlap(rel,query_bundle.query_str)
            if overlap_score>=0.90:
                mentioned_rels.add(rel)
        
        # logging.info(f'####ents:{ents}')
        # logging.info(f'####mentioned_ents:{mentioned_ents}')
        # logging.info(f'####mentioned_rels:{mentioned_rels}')
        
        # 找到与初始实体相邻切重叠度大于 0.9 的实体
        for node in nodes:
            if len(node.node.id_.split('##')) == 2:
                ent, pmid = node.node.id_.split('##')
            else:
                ent  = node.node.id_
                pmid = ent
                
            ents.add(ent)
            
            # logging.info(f"ent: {ent}")

            if (ent not in self.doc2kg) or (pmid not in self.doc2kg) or (len(self.doc2kg[ent])==0):
                continue
            # logging.info(f'####self.doc2kg[ent]: {self.doc2kg[ent]}')
            for triplet in self.doc2kg[ent]:
                h,r,t = triplet
                triplet = [h,r,t]
                if (h in mentioned_ents) and (r in mentioned_rels) and (t not in mentioned_ents):
                    mentioned_ents.add(t)
                if (t in mentioned_ents) and (r in mentioned_rels) and (h not in mentioned_ents):
                    mentioned_ents.add(h)

        mentioned_ents_list = list(mentioned_ents)
        # logging.info(f"明确提及的实体: {mentioned_ents_list}")
        
        for i in range(len(mentioned_ents_list)):
            for j in range(i+1,len(mentioned_ents_list)):
                if (g.has_edge(mentioned_ents_list[i],mentioned_ents_list[j])) or (g.has_edge(mentioned_ents_list[j],mentioned_ents_list[i])):
                    continue
                g.add_edge(mentioned_ents_list[i],mentioned_ents_list[j],rel='cooccurrence',source='query',weight=0.0)
        
        wanted_ents = set(mentioned_ents)
        wanted_rels = set(mentioned_rels)

        # 如果实体数量大于 3 且关系数量大于 0，则提前结束
        early_exit = False
        if (len(wanted_ents)>3 and len(wanted_rels)>0 and self.dataset=='musique'):
            early_exit = True
        early_exit = False

        # 用于找出图 g 中的所有弱连通分量
        wccs = list(nx.connected_components(g))
        # 按长度降序排列
        sorted_wccs = sorted(wccs, key=len,reverse=True)
        cand_ctxs_lists = list()

        logging.info(f"找到了 {len(sorted_wccs)} 个弱连通分量")
        logging.info(f"sorted_wccs: {sorted_wccs}")
        # 处理图的连通分量，并从中提取候选上下文
        for i in range(len(sorted_wccs)):
            if (early_exit) and (len(wanted_ents)==0):
                break
            cand_ctxs_list = []
            wcc = sorted_wccs[i]
            for ent in wcc:
                if ent in wanted_ents:
                    wanted_ents.remove(ent)
            if len(wcc)>1:
                subgraph = g.subgraph(wcc)
                mst = nx.maximum_spanning_tree(subgraph,weight='weight')
                cand_ctx_list = set()
                for cand_edge in mst.edges(data=True):
                    if cand_edge[2]['source']=='query':
                        continue
                    else:
                        if "##" in cand_edge[2]['source']:
                            cand_ctx_list.add(cand_edge[2]['source'].split('##')[1])
                        else:
                            cand_ctx_list.add(cand_edge[2]['source'])
                        
                cand_ctxs_list.append(cand_ctx_list)
            else:
                sorted_edges = sorted(g.subgraph(wcc).edges(data=True),key=lambda x:x[2]['weight'],reverse=True)
                for edge in sorted_edges:
                    if edge[2]['source']=='query':
                        continue
                    else:
                        if "##" in [edge[2]['source'],]:
                            cand_ctxs_list.append([edge[2]['source'],].split('##')[1])
                        else:
                            cand_ctxs_list.append([edge[2]['source'],])
                        break
                    
            cand_ctxs_lists.append(cand_ctxs_list)

        cand_ids_lists = list()
        for cand_ctxs_list in cand_ctxs_lists:
            cand_ids_lists.extend(cand_ctxs_list)
        
        cand_tpts = []
        cand_strs = []
        for cand_ids_list in cand_ids_lists:
            ctx_str = ''
            tpt_str = ''
            for cand_id in cand_ids_list:
                if cand_id in self.doc2kg:
                    cand_ent = cand_id
                    pmid_indx = cand_id
                    ctx_str += self.chunks_index[cand_ent]
                    if (self.use_tpt) and (cand_ent in self.doc2kg) and (pmid_indx in self.doc2kg[cand_ent]) and (len(self.doc2kg[pmid_indx])>0):
                        tpt_str += ', '.join([f'{h} has/is {r} {t}' for h,r,t in self.doc2kg[cand_ent][pmid_indx][:min(len(self.doc2kg[cand_ent][pmid_indx]),3)]])
                        if len(tpt_str)>0:
                            ctx_str = f'{ctx_str} Relational facts: {tpt_str}.'
            cand_strs.append(ctx_str)
            cand_tpts.append(tpt_str)
        
        
        logging.info(f"候选上下文数量: {len(cand_strs)}")
        # logging.info(f"候选上下文: {cand_strs}")
        
        if len(cand_strs)==0:
            scores = self.reranker.compute_score([(query_bundle.query_str,node.node.text) for node in nodes])
            sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
            wanted_nodes = []
            for i in range(min(self.topk,len(sorted_seqs))):
                wanted_nodes.append(nodes[sorted_seqs[i]])
            logging.info(f"最终输出节点数量: {len(wanted_nodes)}")
            return wanted_nodes
        
        # 使用重排序模型（self.reranker）来
        # 评估每个候选上下文与查询之间的相关性，
        # 并根据得分选择最相关的上下文。
        wanted_ctxs = []
        scores = self.reranker.compute_score([(query_bundle.query_str,cand_str) for cand_str in cand_strs])
        # scores = self.reranker.compute_score([(query_bundle.query_str,cand_tpt) for cand_tpt in cand_tpts])
        sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
        for seq in sorted_seqs:
            if len(set(wanted_ctxs)|set(cand_ids_lists[seq]))>self.topk:
                break
            wanted_ctxs.extend(cand_ids_lists[seq])
            wanted_ctxs = list(set(wanted_ctxs))

        if len(wanted_ctxs)<self.topk//2:
            cands = [(query_bundle.query_str,node.node.text,) for node in nodes if node.node.id_ not in wanted_ctxs]
            if len(cands)>0:
                scores = self.reranker.compute_score(cands)
                sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
                for seq in sorted_seqs[:self.topk]:
                    if nodes[seq].node.id_ not in wanted_ctxs:
                        wanted_ctxs.append(nodes[seq].node.id_)
                        if len(wanted_ctxs)>=self.topk:
                            break

        wanted_nodes = []
        for node in nodes:
            if node.node.id_ in wanted_ctxs:
                wanted_nodes.append(node)
        logging.info(f"最终输出节点数量: {len(wanted_nodes)}")
        return wanted_nodes