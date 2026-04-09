import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict

DIFFICULTIES=['easy','medium', 'hard']

@dataclass
class TopicState:
    accuracy: float=0.0
    attempts: int=0
    correct: int=0
    streak: int=0
    current_difficulty: str='easy'
    needs_explanation: bool= False

@dataclass
class StudentModel:
    topics: Dict[str,TopicState] = field(default_factory=dict)
    total_answered: int=0
    session_correct: int=0

    def ensure_topic(self, topic:str):
        if topic not in self.topics:
            self.topics[topic]=TopicState()

    def record_answer(self, topic:str,is_correct:bool):
        self.ensure_topic(topic)
        t=self.topics[topic]
        t.attempts+=1
        self.total_answered+=1

        if is_correct:
            t.correct+=1
            t.streak= max(0, t.streak)+1
            t.needs_explanation=False
            self.session_correct+=1

            #Advance difficulty ad 3 correct answers
            if t.streak>=3:
                idx=DIFFICULTIES.index(t.current_difficulty)
                if idx< len(DIFFICULTIES)-1:
                    t.current_difficulty=DIFFICULTIES[idx+1]
                    t.streak=0
        
        else:
            t.streak= min(0,t.streak)-1
            # Demote difficulty after 2 wrong in a row
            if t.streak<= -2:
                idx=DIFFICULTIES.index(t.current_difficulty)
                if idx>0:
                    t.current_difficulty=DIFFICULTIES[idx-1]
                t.streak=0
                t.needs_explanation=True

        t.accuracy=t.correct/t.attempts if t.attempts>0 else 0.0
    
    def select_next_topic(self, available_topics:list)-> str:
        """ Pick The Topic with the lowest accuarcy (Whicnh needs the most work)."""
        self.ensure_topic(available_topics[0])
        for topic in available_topics:
            self.ensure_topic(topic)
        return min(available_topics, key=lambda t:(
            self.topics[t].accuracy,
            -self.topics[t].attempts
        ))
    
    def get_mastery_summary(self)-> dict:
        return{
            topic: {
               "accuracy":f"{state.accuracy:.0%}" ,
               "difficulty":state.current_difficulty,
               "attempts":state.attempts
            }
            for topic, state in self.topics.items()
        }
    
    def save(self, path="student_model.json"):
        with open(path,"w") as f:
            data ={k: asdict(v) for k,v in self.topics.items() }
            json.dump({"topics":data, "total_answered": self.total_answered,
                       "session_correct":self.session_correct},f , indent=2)
            
    @classmethod
    def load(cls,path="student_model.json")-> "StudentModel":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            raw=json.load(f)
        model= cls()
        model.total_answered= raw.get("total_answered",0)
        model.session_correct= raw.get("session_correct", 0)
        for topic, state_dict in raw.get("topics", {}).items():
            model.topics[topic]=TopicState(**state_dict)
        return model
