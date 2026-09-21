"""AutoReport —— 多智能体自动研报系统。

架构分四层：
1. data_ingestion  研报数据获取层（采集/解析/入库）
2. retrieval       检索层（混合检索 + 引用溯源）
3. agents          多智能体层（LangGraph 编排：主管/检索/工具/写作）
4. evaluation      评测层（LangSmith 评测闭环 + 本地降级）

版权声明：本项目仅用于个人学习与研究，研报版权归原机构所有，不二次分发、不商用。
"""

__version__ = "0.1.0"
