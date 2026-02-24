export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  imageUrl?: string;
  timestamp: Date;
};

export type ChatSession = {
  id: string;
  title: string;
  lastMessage: string;
};

export const mockSessions: ChatSession[] = [
  {
    id: "1",
    title: "血压偏高咨询",
    lastMessage: "建议您定期监测血压...",
  },
  {
    id: "2",
    title: "体检报告解读",
    lastMessage: "您的各项指标总体正常...",
  },
  {
    id: "3",
    title: "糖尿病用药咨询",
    lastMessage: "二甲双胍的服用注意事项...",
  },
];
