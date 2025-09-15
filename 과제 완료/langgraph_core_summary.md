# LangGraph 핵심 개념 정리

## 1. TypedDict & Annotated
```python
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from typing import List
```

- **TypedDict**: 변수의 타입을 명시하여 타입 불일치를 확인할 수 있음.
  ```python
  class Person(TypedDict):
      name: str
      age: int
  ```

- **Annotated**: 타입에 "추가 메타데이터(설명, 제약 조건 등)"를 줄 수 있음.
  ```python
  name: Annotated[str, "사용자 이름"]
  age: Annotated[int, "사용자 나이 (0-150)"]
  ```

- **Pydantic 예시**
  ```python
  class Employee(BaseModel):
      id: Annotated[int, Field(..., description="직원 ID")]
      name: Annotated[str, Field(..., min_length=3, max_length=50, description="이름")]
      age: Annotated[int, Field(gt=18, lt=65, description="나이 (19-64세)")]
      salary: Annotated[float, Field(gt=0, lt=10000, description="연봉 (단위: 만원, 최대 10억)")]
      skills: Annotated[List[str], Field(min_items=1, max_items=10, description="보유 기술 (1-10개)")]
  ```

---

## 2. add_messages
- **정의**: LangGraph에서 `messages` 필드를 누적 관리하기 위한 유틸리티 함수  
- 단순 리스트 append가 아니라, 대화 맥락을 올바르게 이어가도록 설계됨.

```python
class MyData(TypedDict):
    messages: Annotated[list, add_messages]
```

---

## 3. LangGraph 기본 구조

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_teddynote.graphs import visualize_graph
```

### (1) 상태(State) 정의
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### (2) 노드(Node) 정의
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
```

### (3) 그래프(Graph) 정의 및 컴파일
```python
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
visualize_graph(graph)
```

### (4) 실행
```python
question = "서울의 유명한 맛집 TOP 10 추천해줘"

for event in graph.stream({"messages": [("user", question)]}):
    for value in event.values():
        print(value["messages"][-1].content)
```

---

## 4. 조건부 엣지 (Conditional Edge)
- 2개 이상의 edge를 가질 수 있으며, 조건에 따라 다른 Node로 상태 전달.

---

## 5. MemorySaver
- LangGraph에서 실행 기록을 저장할 때 사용됨.
- 컴파일 시 옵션으로 전달:
  ```python
  graph = graph_builder.compile(checkpointer=memory)
  ```

---

## 6. RunnableConfig
- **recursion_limit**: 노드 방문의 최대 깊이 (무한 루프 방지)  
- **configurable**: 스레드 ID 설정  
- **tags**: 태그를 달아 실행 구분 가능  

```python
graph.get_state(config)
# 그래프의 스냅샷을 불러와 config, values, next(다음 노드) 값 확인 가능
```

---

## 7. Stream & 관련 옵션
- **stream**: 실행 도중 이벤트 단위로 출력

- **입력 (input)**: 실행 시 최초 데이터  
  ```python
  input = State(dummy_data="테스트 문자열", messages=[("user", question)])
  ```

- **출력 (output_keys)**: 기본은 `["messages"]`  

- **interrupt_before / interrupt_after**: 특정 노드 직전·직후에 실행 멈춤  

- **stream_mode**:
  - `values`: 각 노드 실행이 끝날 때 반환된 값 단위 이벤트
  - `updates`: 상태(state)가 변경될 때마다 이벤트
  - `messages`: 메시지 객체 단위 스트리밍
  - 혼합 가능 → `stream_mode=["values","updates"]`

---

## 8. Human-in-the-loop
- LangChain/LangGraph 실행 과정에서 사람이 직접 개입하여 승인·수정 가능
- 본 예제에서는:
  - `interrupt_before`, `get_state_history`, `update_state`를 사용하여 특정 지점에서 멈춤
  - 사람이 개입해 상태를 업데이트한 뒤 다시 실행
- (8강에서는 정석적인 Human-in-the-loop 예제가 제공됨)

---

# ✅ 최종 정리
위 내용은 LangGraph 핵심 개념을 **TypedDict/Annotated → add_messages → Graph 구조 → Edge → Memory → Config → Stream → Human-in-the-loop** 순서로 정리하였으며, 몇몇 설명은 보다 정확하게 수정하였습니다.
