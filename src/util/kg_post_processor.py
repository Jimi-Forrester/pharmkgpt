import logging
import networkx as nx

from typing import List,Dict,Optional,Set
from FlagEmbedding import FlagReranker
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher
dispatcher = get_dispatcher(__name__)

from builtins import print as _print
from sys import _getframe
def print(*arg, **kw):
    s = f'Line {_getframe(1).f_lineno}'
    return _print(f"Func {__name__} - {s}", *arg, **kw)


import re

def has_pmid(text):
    """
    判断字符串中是否包含 PMID。

    Args:
        text: 要判断的字符串。

    Returns:
        True: 如果字符串中包含 PMID，否则返回 False。
    """
    pattern = r'\bpmid:\s*\d+\b'  # 匹配 "PMID:" 后面跟着数字
    match = re.search(pattern, text, re.IGNORECASE) # 忽略大小写
    return bool(match)

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
)

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
        """Postprocess nodes"""
        entity_order = {}
        sorted_nodes = []
        logging.info("----------------NaivePostprocessor------------")
        logging.info(f">>初始节点: {len(nodes)}")
        logging.info(f"排序前的节点: {[i.id_ for i in nodes]}")
        for i, node in enumerate(nodes):
            node_id = node.node.id_
            ent = node_id
            ctx_seq = i
            if ent not in entity_order:
                entity_order[ent] = len(entity_order)
            sorted_nodes.append((ent,ctx_seq,node))
        sorted_nodes.sort(key=lambda x:(entity_order[x[0]],x[1]))
        sorted_nodes = [node for _,_,node in sorted_nodes]
        logging.info(f"排序后的节点: {[i.id_ for i in sorted_nodes]}")

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
            
        logging.info(f">>初始节点: {len(sorted_nodes)}")
        return sorted_nodes
    
class KGRetrievePostProcessor(BaseNodePostprocessor):
    """
    继承BaseNodePostprocessor
    直接找到与query相关的实体,然后找到与这些实体相关的文本
    """

    dataset: str = Field
    ents: Set[str] = Field
    doc2kg: Dict[str,Dict[str, List[tuple]]] = Field
    chunks_index: Dict[str,Dict[str,str]] = Field
    hops: int = Field

    @classmethod
    def class_name(cls) -> str:
        return "KGRetrievePostprocessor"
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        
        logging.info("------------KGRetrievePostProcessor------------")
        top_k = len(nodes)
        retrieved_ids = set()
        retrieved_ents = set()

        related_ents = set()
        textid2score = dict()
        
        logging.info(">> 拿到初始节点信息")
        highly_related_ents = set()
        ent_count = dict()
        ent_score = dict()
        
        
        
        for i in range(len(nodes)):
            node = nodes[i]
            node_id = node.node.id_
            retrieved_ids.add(node_id)
            textid2score[node_id] = node.score
            entity = node_id
            seq_str = node_id

            if (i<(top_k//2)) and (entity in retrieved_ents):
                highly_related_ents.add(entity)
            retrieved_ents.add(entity)

            if entity not in ent_count:
                ent_count[entity] = 0
                ent_score[entity] = 0.0
            ent_count[entity] += 1
            ent_score[entity] += node.score


        sorted_ents = sorted(ent_count.keys(),key=lambda x:(ent_score[x]/ent_count[x]),reverse=True)
        for i in range(min(2,len(sorted_ents))):
            highly_related_ents.add(sorted_ents[i])

        logging.info(f"** 初始节点: {retrieved_ents}")
        logging.info(f"** 初始节点得分: {textid2score}")
        logging.info(f"** highly_related_ents: {highly_related_ents}")
        
        logging.info(">> 检索初始节点相邻的节点，作为additional_ents")
        additional_ents = set()
        for node in nodes:
            node_id = node.node.id_
            entity = node_id
            seq_str = node_id
            idx_seq_str = node_id

            if (entity not in self.doc2kg) or (idx_seq_str not in self.doc2kg[entity]):
                continue
            for triplet in self.doc2kg[entity][idx_seq_str]:
                logging.info(f"** 检索到triplet: {triplet}")
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
        logging.info(f"** additional_ents: {additional_ents}")
        # ----------------------多跳扩展----------------------

        # logging.info(f">> 开始多跳扩展")
        # for hop in range(self.hops):
        #     related_ents = related_ents.union(additional_ents)
        #     temp_ents = set(additional_ents)
        #     additional_ents = set()
        #     if 'delirium' in temp_ents:
        #         temp_ents.remove('delirium') # 中心节点，检索搭配太多
            
        #     for ent in temp_ents:
        #         if (ent not in self.doc2kg) or (len(self.doc2kg[ent])==0):
        #             continue
        #         for idx_seq_str in self.doc2kg[ent]:
                    
        #             if len(self.doc2kg[ent][idx_seq_str])==0:
        #                 continue
        #             ctx_id = idx_seq_str
        #             if ctx_id in retrieved_ids:
        #                 continue
                    
        #             for triplet in self.doc2kg[ent][idx_seq_str]:
        #                 logging.info(f"** hops {hop}: {triplet}")
        #                 h,r,t = triplet
        #                 if (h in self.ents) and (h not in related_ents):
        #                 # if h in additional_ents:
        #                     additional_ents.add(h)
        #                     if h not in ent_count:
        #                         ent_count[h] = 0
        #                         ent_score[h] = 0.0
        #                     ent_count[h] += 1
        #                     ent_score[h] += float((ent_score[ent])/(ent_count[ent]))

        #                 # if t in additional_ents:
        #                 if (t in self.ents) and (t not in related_ents):
        #                     additional_ents.add(t)
        #                     if t not in ent_count:
        #                         ent_count[t] = 0
        #                         ent_score[t] = 0.0
        #                     ent_count[t] += 1
        #                     ent_score[t] += float((ent_score[ent])/(ent_count[ent]))

        
        related_ents = related_ents.union(additional_ents)
        logging.info(f"** 多跳扩展的additional_ents: {len(related_ents)}")
        additional_ids = set()

        # ----------------------添加相关实体的文本----------------------
        logging.info(">> 扩充后的实体中检索的实体")
        avg_score = float(sum([node.score for node in nodes])/len(nodes))
        retrieved_ents = retrieved_ents-highly_related_ents
        for ent in (related_ents-retrieved_ents):
            if (ent not in self.chunks_index) or (len(self.chunks_index[ent])==0) or (ent not in self.doc2kg):
                continue
            
            for idx_seq_str in self.chunks_index[ent]:
                # logging.info(f"** idx_seq_str: {idx_seq_str}")
                ctx_id = idx_seq_str
                
                # ctx_id = ent
                if ctx_id in retrieved_ids:
                    continue
                
                # logging.info(f"** ctx_id: {ctx_id}")
                additional_ids.add(ctx_id)
                textid2score[ctx_id] = 0.0
                if ent in ent_score:
                    logging.info(f"{ctx_id}:{(ent_score[ent]+avg_score)/(ent_count[ent]+1)}")
                    textid2score[ctx_id] += (ent_score[ent]+avg_score)/(ent_count[ent]+1)

        logging.info(f"** additional_ids: {len(additional_ids)}")
        
        added_nodes = []
        for ctx_id in additional_ids:
            ent = ctx_id
            seq_str = ctx_id
            idx_seq_str = seq_str
            if ent in self.chunks_index:
                if idx_seq_str in self.chunks_index[ent]:
                    ctx_text =  self.chunks_index[ent][idx_seq_str]
                    node = TextNode(id_=ctx_id,text=ctx_text)
                    node = NodeWithScore(node=node, score=textid2score[ctx_id])
                    added_nodes.append(node)
        added_nodes = sorted(added_nodes,key=lambda x:x.score,reverse=True)
        added_nodes = added_nodes[:10]
        
        logging.info(f"**added_nodes: {[node.id_ for node in added_nodes]}")
        
        nodes = nodes+added_nodes
        nodes = [node for node in nodes if node.id_[:4].lower() == "pmid"]
        logging.info(f">>>>>> KGRetrieve output: {len(nodes)}")
        logging.info(f">>>>>> KGRetrieve output nodes: {[node.id_ for node in nodes if nodes]}")
        return nodes

class GraphFilterPostProcessor(BaseNodePostprocessor):

    """KnowledgeGraph-based Node processor."""
    dataset: str = Field
    topk: int = Field
    use_tpt: bool = Field
    ents: Set[str] = Field
    doc2kg: Dict[str,Dict[str, List[tuple]]] = Field
    chunks_index: Dict[str,Dict[str,str]] = Field
    reranker: FlagReranker = Field

    @classmethod
    def class_name(cls) -> str:
        return "GraphFilterPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes"""
        logging.info("------------------GraphFilterPostprocessor------------------")
        ents = set()
        rels = set()

        g = nx.MultiGraph()

        logging.info('>> 写入初始节点')
        for node in nodes:
            ent = node.node.id_
            ents.add(ent)

        logging.info(">> 开始构建子图")
        for node in nodes:
            ent = node.node.id_
            idx_seq_str = node.node.id_
            if (ent not in self.doc2kg) or (idx_seq_str not in self.doc2kg[ent]) or (len(self.doc2kg[ent][idx_seq_str])==0):
                continue
            for triplet in self.doc2kg[ent][idx_seq_str]:
                h,r,t = triplet
                h = h.strip()
                r = r.strip()
                t = t.strip()
                triplet = [h,r,t]
                ents.add(h)
                ents.add(t)
                rels.add(r)
                g.add_edge(h,t,rel=r,source=node.node.id_,weight=node.score)

        logging.info(">> 找到与初始实体相邻切重叠度大于 0.9 的实体")
        mentioned_ents = set()
        mentioned_rels = set()

        for ent in ents:
            overlap_score = ngram_overlap(ent,query_bundle.query_str)
            if overlap_score>=0.90:
                mentioned_ents.add(ent)
        for rel in rels:
            overlap_score = ngram_overlap(rel,query_bundle.query_str)
            if overlap_score>=0.90:
                mentioned_rels.add(rel)

        for node in nodes:
            ent = node.node.id_
            idx_seq_str = node.node.id_
            if (ent not in self.doc2kg) or (idx_seq_str not in self.doc2kg[ent]) or (len(self.doc2kg[ent][idx_seq_str])==0):
                continue
            for triplet in self.doc2kg[ent][idx_seq_str]:
                h,r,t = triplet
                triplet = [h,r,t]
                if (h in mentioned_ents) and (r in mentioned_rels) and (t not in mentioned_ents):
                    mentioned_ents.add(t)
                if (t in mentioned_ents) and (r in mentioned_rels) and (h not in mentioned_ents):
                    mentioned_ents.add(h)

        mentioned_ents_list = list(mentioned_ents)
        for i in range(len(mentioned_ents_list)):
            for j in range(i+1,len(mentioned_ents_list)):
                if (g.has_edge(mentioned_ents_list[i],mentioned_ents_list[j])) or (g.has_edge(mentioned_ents_list[j],mentioned_ents_list[i])):
                    continue
                g.add_edge(mentioned_ents_list[i],mentioned_ents_list[j],rel='cooccurrence',source='query',weight=0.0)
        
        wanted_ents = set(mentioned_ents)
        wanted_rels = set(mentioned_rels)
        logging.info(f"**mentioned_ents:{len(mentioned_ents)}")
        
        early_exit = False
        if (len(wanted_ents)>3 and len(wanted_rels)>0 and self.dataset=='musique'):
            early_exit = True
        early_exit = False

        logging.info(">> 开始找图中最大的连通分量的节点")
        wccs = list(nx.connected_components(g))
        sorted_wccs = sorted(wccs,key=len,reverse=True)
        
        cand_ctxs_lists = list()

        # for i in range(len(sorted_wccs)):
        #     if (early_exit) and (len(wanted_ents)==0):
        #         break
        #     cand_ctxs_list = []
        #     wcc = sorted_wccs[i]
        #     for ent in wcc:
        #         if ent in wanted_ents:
        #             wanted_ents.remove(ent)
        #     if len(wcc)>1:
        #         subgraph = g.subgraph(wcc)
        #         mst = nx.maximum_spanning_tree(subgraph,weight='weight')
        #         cand_ctx_list = []
        #         for cand_edge in mst.edges(data=True):
        #             if cand_edge[2]['source']=='query':
        #                 continue
        #             else:
        #                 cand_ctx_list.append(cand_edge[2]['source'])
        #         cand_ctxs_list.append(cand_ctx_list)
        #     else:
        #         sorted_edges = sorted(g.subgraph(wcc).edges(data=True),key=lambda x:x[2]['weight'],reverse=True)
        #         for edge in sorted_edges:
        #             if edge[2]['source']=='query':
        #                 continue
        #             else:
        #                 cand_ctxs_list.append([edge[2]['source'],])
        #                 break
        #     cand_ctxs_lists.append(cand_ctxs_list)

        min_weight_threshold = 0.5 # 设置权重阈值
        min_degree = 1  # 设置最小度数
        num_wccs_to_consider = 1
        logging.info(f"**len(sorted_wccs):{len(sorted_wccs)}")
        for i in range(min(num_wccs_to_consider, len(sorted_wccs))): 
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
                cand_ctx_list = []
                for u, v, data in mst.edges(data=True):
                    if data['source']=='query':
                        continue

                    source_node = data['source']
                    # 权重判断
                    if data['weight'] < min_weight_threshold:
                        continue

                    # 度数判断
                    degree = mst.degree(source_node)
                    if degree < min_degree:
                        continue

                    cand_ctx_list.append(source_node)
                cand_ctxs_list.append(cand_ctx_list)
            else:
                sorted_edges = sorted(g.subgraph(wcc).edges(data=True),key=lambda x:x[2]['weight'],reverse=True)
                for edge in sorted_edges:
                    if edge[2]['source']=='query':
                        continue
                    else:
                        cand_ctxs_list.append([edge[2]['source'],])
                        break
            cand_ctxs_lists.append(cand_ctxs_list)
            
        cand_ids_lists = list()
        for cand_ctxs_list in cand_ctxs_lists:
            cand_ids_lists.extend(cand_ctxs_list)
        
        logging.info(f"**得到候选节点：{cand_ids_lists}")
        
        cand_tpts = []
        cand_strs = []
        for cand_ids_list in cand_ids_lists:
            ctx_str = ''
            tpt_str = ''
            for cand_id in cand_ids_list:
                cand_ent = cand_id
                idx_seq_str = cand_id
                for idx_seq_str in self.chunks_index[cand_ent]:
                    ctx_str += self.chunks_index[cand_ent][idx_seq_str]
                    
                if (self.use_tpt) and (cand_ent in self.doc2kg) and (idx_seq_str in self.doc2kg[cand_ent]) and (len(self.doc2kg[cand_ent][idx_seq_str])>0):
                    for idx_seq_str in self.chunks_index[cand_ent]:
                        tpt_str += ', '.join([f'{h} has/is {r} {t}' for h,r,t in self.doc2kg[cand_ent][idx_seq_str][:min(len(self.doc2kg[cand_ent][idx_seq_str]),3)]])
                        if len(tpt_str)>0:
                            ctx_str = f'{ctx_str} Relational facts: {tpt_str}.'
            cand_strs.append(ctx_str)
            cand_tpts.append(tpt_str)
        
        logging.info(">> 候选上下文内容准备好")

        if len(cand_strs)==0:
            logging.info(">> 没有候选节点，重排列原始节点")
            scores = self.reranker.compute_score([(query_bundle.query_str,node.node.text) for node in nodes])
            sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
            wanted_nodes = [nodes[sorted_seqs[i]] for i in range(min(self.topk,len(sorted_seqs)))]
            return wanted_nodes
        
        
        logging.info(">> 重排候选节点")
        wanted_ctxs = []
        scores = self.reranker.compute_score([(query_bundle.query_str,cand_str) for cand_str in cand_strs])
        # scores = self.reranker.compute_score([(query_bundle.query_str,cand_tpt) for cand_tpt in cand_tpts])
        sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
        for seq in sorted_seqs:
            if len(set(wanted_ctxs)|set(cand_ids_lists[seq]))>self.topk:
                break
            wanted_ctxs.extend(cand_ids_lists[seq])
            wanted_ctxs = list(set(wanted_ctxs))

        logging.info(f"** wanted_ctxs: {wanted_ctxs}")
        
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
                
        logging.info(f">> GraphFilter output: {len(wanted_nodes)}")
        logging.info(f">> GraphFilter output nodes: {[i.id_ for i in wanted_nodes]}")
        return wanted_nodes