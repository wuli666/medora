import { useState, useRef, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sparkles, Send, Paperclip, Plus, MessageSquare, Menu, X,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import {
  type ChatMessage,
  type ChatSession,
  mockSessions,
} from "@/lib/mock-data";

/* ── Typewriter hook ── */
function useTypewriter(text: string, speed = 15) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);

  useEffect(() => {
    setDisplayed("");
    setDone(false);
    let i = 0;
    const iv = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(iv);
        setDone(true);
      }
    }, speed);
    return () => clearInterval(iv);
  }, [text, speed]);

  return { displayed, done };
}

/* ── Typing bubble ── */
function AssistantBubble({ content, isLatest }: { content: string; isLatest: boolean }) {
  const { displayed, done } = useTypewriter(content, isLatest ? 12 : 0);
  const show = isLatest && !done ? displayed : content;

  return (
    <div className="glass-card border-sky-200/70 bg-sky-50/55 p-5 max-w-[85%] md:max-w-[70%] prose prose-sm prose-neutral dark:prose-invert max-w-none shadow-[0_10px_24px_hsla(var(--gradient-start)/0.10)]">
      <ReactMarkdown>{show}</ReactMarkdown>
      {isLatest && !done && (
        <span className="inline-block w-2 h-4 bg-primary/60 rounded-sm animate-pulse ml-0.5" />
      )}
    </div>
  );
}

type StageItem = {
  key: string;
  label: string;
  status: string;
  content: string;
};

type MultiAgentResponse = {
  run_id: string;
  summary: string;
  planner: string;
  tool: string;
  search: string;
  reflect_verify: string;
  stages: StageItem[];
};

type UiMessage = ChatMessage & {
  stages?: StageItem[];
};
type LoadingStage = { key: string; label: string };

const QUICK_PROMPTS = [
  "请用通俗语言解释这份检查报告",
  "帮我梳理复诊前需要关注的三个重点",
  "根据目前情况给我一周的健康管理建议",
];

const formatTime = (date: Date) =>
  new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);

const STAGE_CLASS: Record<string, string> = {
  quick_router: "border-indigo-200 bg-indigo-50/60",
  tooler: "border-sky-200 bg-sky-50/60",
  searcher: "border-violet-200 bg-violet-50/60",
  planner: "border-cyan-200 bg-cyan-50/60",
  reflector: "border-amber-200 bg-amber-50/60",
  summarize: "border-emerald-200 bg-emerald-50/60",
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 120000;
const DEFAULT_LOADING_FLOW: LoadingStage[] = [{ key: "quick_router", label: "意图识别" }];

const buildStagesFromLegacy = (payload: Partial<MultiAgentResponse>): StageItem[] => {
  if (payload.stages && payload.stages.length > 0) return payload.stages;
  return [
    {
      key: "tooler",
      label: "病历/影像解析",
      status: payload.tool ? "done" : "skipped",
      content: payload.tool || "",
    },
    {
      key: "searcher",
      label: "医学检索补充",
      status: payload.search ? "done" : "skipped",
      content: payload.search || "",
    },
    {
      key: "planner",
      label: "管理计划生成",
      status: payload.planner ? "done" : "skipped",
      content: payload.planner || "",
    },
    {
      key: "reflector",
      label: "一致性校验",
      status: payload.reflect_verify ? "done" : "skipped",
      content: payload.reflect_verify || "",
    },
    {
      key: "summarize",
      label: "患者摘要",
      status: payload.summary ? "done" : "skipped",
      content: payload.summary || "",
    },
  ];
};

/* ── Main page ── */
const Chat = () => {
  const [sessions] = useState<ChatSession[]>(mockSessions);
  const [messages, setMessages] = useState<UiMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "您好，我是 **MedInsight 智能医疗助手**。\n\n我可以帮您解读病历、分析影像、并生成随访建议。您可以先上传资料，再直接提问。",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingFlow, setLoadingFlow] = useState<LoadingStage[]>([]);
  const [loadingStageIdx, setLoadingStageIdx] = useState(0);
  const [loadingVisibleCount, setLoadingVisibleCount] = useState(1);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }, 50);
  }, []);

  useEffect(scrollToBottom, [messages, scrollToBottom]);

  const updateTimelineFromStages = (stages: StageItem[]) => {
    if (!stages || stages.length === 0) return;
    const flow = stages.map((s) => ({ key: s.key, label: s.label }));
    setLoadingFlow(flow);

    let runningIdx = stages.findIndex((s) => s.status === "running");
    if (runningIdx < 0) {
      runningIdx = stages.findIndex((s) => s.status === "pending");
      if (runningIdx > 0) runningIdx -= 1;
      if (runningIdx < 0) runningIdx = stages.length - 1;
    }
    setLoadingStageIdx(Math.max(0, runningIdx));

    let visible = 1;
    for (let i = 0; i < stages.length; i += 1) {
      if (stages[i].status !== "pending") visible = i + 1;
      else break;
    }
    setLoadingVisibleCount(Math.max(1, visible));
  };

  useEffect(() => () => eventSourceRef.current?.close(), []);

  const handleSend = async () => {
    const text = input.trim();
    if (!text && !previewImage) return;
    const outboundText = text || "请分析这张影像";

    const userMsg: UiMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: outboundText,
      imageUrl: previewImage ?? undefined,
      timestamp: new Date(),
    };
    setMessages((m) => [...m, userMsg]);
    setIsLoading(true);
    setLoadingFlow(DEFAULT_LOADING_FLOW);
    setLoadingStageIdx(0);
    setLoadingVisibleCount(1);

    try {
      const runId = crypto.randomUUID();
      eventSourceRef.current?.close();
      const sse = new EventSource(`${API_BASE}/api/multi-agent/events/${runId}`);
      eventSourceRef.current = sse;
      sse.addEventListener("run_started", (event) => {
        const payload = JSON.parse((event as MessageEvent).data);
        updateTimelineFromStages(payload.stages || []);
      });
      sse.addEventListener("snapshot", (event) => {
        const payload = JSON.parse((event as MessageEvent).data);
        updateTimelineFromStages(payload.stages || []);
      });
      sse.addEventListener("stage_update", (event) => {
        const payload = JSON.parse((event as MessageEvent).data);
        updateTimelineFromStages(payload.stages || []);
      });
      sse.addEventListener("run_completed", () => sse.close());
      sse.addEventListener("run_failed", () => sse.close());
      sse.onerror = () => {
        sse.close();
      };

      const formData = new FormData();
      formData.append("run_id", runId);
      formData.append("patient_text", outboundText);
      if (selectedImageFile) {
        formData.append("image", selectedImageFile);
      }

      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      const response = await fetch(`${API_BASE}/api/multi-agent/run`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      window.clearTimeout(timeoutId);

      const raw = await response.text();
      let payload: Partial<MultiAgentResponse> & { detail?: string } = {};
      try {
        payload = raw ? JSON.parse(raw) : {};
      } catch {
        payload = { detail: raw };
      }
      if (!response.ok) {
        throw new Error(payload.detail || `请求失败: ${response.status}`);
      }

      const aiMsg: UiMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: payload.summary || "未返回总结内容",
        timestamp: new Date(),
        stages: buildStagesFromLegacy(payload),
      };
      setMessages((m) => [...m, aiMsg]);
      setInput("");
      setPreviewImage(null);
      setSelectedImageFile(null);
      if (fileRef.current) fileRef.current.value = "";
    } catch (error) {
      const aiMsg: UiMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          error instanceof DOMException && error.name === "AbortError"
            ? `请求超时（>${REQUEST_TIMEOUT_MS / 1000}s），请稍后重试。`
            : `请求失败：${error instanceof Error ? error.message : "未知错误"}`,
        timestamp: new Date(),
      };
      setMessages((m) => [...m, aiMsg]);
    } finally {
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
      setIsLoading(false);
      setLoadingFlow([]);
      setLoadingStageIdx(0);
      setLoadingVisibleCount(1);
    }
  };

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedImageFile(file);
    const reader = new FileReader();
    reader.onload = () => setPreviewImage(reader.result as string);
    reader.readAsDataURL(file);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleQuickPrompt = (text: string) => {
    setInput(text);
  };

  return (
    <div className="h-screen flex relative overflow-hidden">
      {/* Background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute top-[-15%] left-[-5%] w-[400px] h-[400px] rounded-full bg-primary/[0.08] blur-[100px]" />
        <div className="absolute bottom-[-10%] right-[-5%] w-[350px] h-[350px] rounded-full bg-accent/[0.08] blur-[100px]" />
      </div>

      {/* Sidebar overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            className="fixed inset-0 bg-foreground/20 backdrop-blur-sm z-40 md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            className="fixed md:relative z-50 top-0 left-0 h-full w-72 glass-card rounded-none border-r flex flex-col"
            initial={{ x: -288 }}
            animate={{ x: 0 }}
            exit={{ x: -288 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
          >
            <div className="p-4 flex items-center justify-between border-b border-border/50">
              <span className="font-semibold text-sm">历史会话</span>
              <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(false)} className="rounded-xl">
                <X className="w-4 h-4" />
              </Button>
            </div>
            <ScrollArea className="flex-1 p-3">
              <Button variant="ghost" className="w-full justify-start gap-2 mb-2 rounded-xl text-sm">
                <Plus className="w-4 h-4" /> 新会话
              </Button>
              {sessions.map((s) => (
                <button
                  key={s.id}
                  className="w-full text-left p-3 rounded-xl hover:bg-muted/60 transition-colors mb-1"
                >
                  <div className="flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-muted-foreground shrink-0" />
                    <div className="min-w-0">
                      <p className="text-sm font-medium truncate">{s.title}</p>
                      <p className="text-xs text-muted-foreground truncate">{s.lastMessage}</p>
                    </div>
                  </div>
                </button>
              ))}
            </ScrollArea>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="glass-nav px-4 h-14 flex items-center gap-3 shrink-0">
          <Button variant="ghost" size="icon" className="rounded-xl" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <Menu className="w-5 h-5" />
          </Button>
          <Link to="/" className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg gradient-bg flex items-center justify-center">
              <Sparkles className="w-3.5 h-3.5 text-primary-foreground" />
            </div>
            <span className="font-semibold text-sm">MedInsight</span>
          </Link>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
          <div className="max-w-4xl mx-auto space-y-4">
          {messages.map((msg, idx) => (
            <motion.div
              key={msg.id}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {msg.role === "user" ? (
                <div className="max-w-[85%] md:max-w-[70%] space-y-2">
                  {msg.imageUrl && (
                    <div className="rounded-2xl overflow-hidden border border-border/50 glass-card">
                      <img src={msg.imageUrl} alt="上传的图片" className="max-h-64 object-contain w-full" />
                    </div>
                  )}
                  <div className="gradient-bg text-primary-foreground p-4 rounded-2xl rounded-br-md shadow-[0_10px_24px_hsla(var(--gradient-start)/0.28)]">
                    <p className="text-sm leading-relaxed">{msg.content}</p>
                  </div>
                  <p className="text-[11px] text-muted-foreground text-right pr-1">
                    {formatTime(msg.timestamp)}
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  <AssistantBubble
                    content={msg.content}
                    isLatest={idx === messages.length - 1}
                  />
                  {msg.stages && msg.stages.length > 0 && (
                    <details className="pl-1">
                      <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground transition-colors">
                        查看阶段详情
                      </summary>
                      <div className="space-y-2 mt-2">
                        {msg.stages.map((stage) => (
                          <div
                            key={`${msg.id}-${stage.key}`}
                            className={`rounded-xl border p-3 ${STAGE_CLASS[stage.key] || "border-muted bg-muted/50"}`}
                          >
                            <p className="text-xs font-semibold text-foreground/80">
                              {stage.label}
                            </p>
                            <p className="text-xs text-muted-foreground mt-1 max-h-24 overflow-auto">
                              {stage.content || "无阶段输出"}
                            </p>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  <p className="text-[11px] text-muted-foreground pl-1">
                    {formatTime(msg.timestamp)}
                  </p>
                </div>
              )}
            </motion.div>
          ))}

          {messages.length === 1 && (
            <motion.div
              className="flex justify-start"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="glass-card p-3 rounded-2xl max-w-[92%] md:max-w-[75%]">
                <p className="text-xs text-muted-foreground mb-2">你可以这样开始：</p>
                <div className="flex flex-wrap gap-2">
                  {QUICK_PROMPTS.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      className="text-xs px-3 py-1.5 rounded-full bg-muted/70 hover:bg-muted transition-colors"
                      onClick={() => handleQuickPrompt(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {isLoading && (
            <motion.div className="flex justify-start" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="glass-card p-4 min-w-[340px]">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    {[0, 1, 2].map((i) => (
                      <span
                        key={i}
                        className="w-2 h-2 rounded-full bg-primary/50 animate-bounce"
                        style={{ animationDelay: `${i * 0.15}s` }}
                      />
                    ))}
                  </div>
                  <span className="text-sm text-muted-foreground">
                    正在执行：{loadingFlow[loadingStageIdx]?.label || "等待后端阶段回传"}
                  </span>
                </div>
                <div className="mt-4 space-y-2">
                  {loadingFlow.slice(0, loadingVisibleCount).map((stage, idx) => {
                    const isDone = idx < loadingStageIdx;
                    const isCurrent = idx === loadingStageIdx;
                    const isLast = idx === loadingFlow.length - 1;
                    return (
                      <div key={stage.key} className="flex items-start gap-3">
                        <div className="relative flex flex-col items-center">
                          <span
                            className={[
                              "h-3 w-3 rounded-full border transition-all",
                              isCurrent
                                ? "bg-primary border-primary shadow-[0_0_0_4px_hsla(var(--ring)/0.2)]"
                                : isDone
                                  ? "bg-emerald-500 border-emerald-500"
                                  : "bg-background/80 border-border",
                            ].join(" ")}
                          />
                          {!isLast && (
                            <span
                              className={[
                                "mt-1 w-px h-6",
                                isDone ? "bg-emerald-400/80" : "bg-border/70",
                              ].join(" ")}
                            />
                          )}
                        </div>
                        <div
                          className={[
                            "text-xs px-2 py-1 rounded-md border transition-colors",
                            STAGE_CLASS[stage.key] || "border-muted bg-muted/50",
                            isCurrent ? "text-foreground font-medium" : "text-muted-foreground",
                          ].join(" ")}
                        >
                          {stage.label}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}
          </div>
        </div>

        {/* Image preview */}
        <AnimatePresence>
          {previewImage && (
            <motion.div
              className="px-4 pb-2"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
            >
              <div className="glass-card p-2 inline-flex items-center gap-2 rounded-xl">
                <img src={previewImage} alt="预览" className="h-16 rounded-lg object-cover" />
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-lg h-8 w-8"
                  onClick={() => {
                    setPreviewImage(null);
                    setSelectedImageFile(null);
                    if (fileRef.current) fileRef.current.value = "";
                  }}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input */}
        <div className="p-4 shrink-0">
          <div className="glass-card p-3 flex items-end gap-2 max-w-3xl mx-auto border-white/45 shadow-[0_10px_40px_hsla(var(--glass-shadow))]">
            <input type="file" ref={fileRef} className="hidden" accept="image/*" onChange={handleFile} />
            <Button
              variant="ghost"
              size="icon"
              className="rounded-xl shrink-0 text-muted-foreground hover:text-foreground"
              onClick={() => fileRef.current?.click()}
            >
              <Paperclip className="w-5 h-5" />
            </Button>
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入您的问题，或上传病历/影像..."
              className="min-h-[44px] max-h-32 resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 text-sm leading-relaxed"
              rows={1}
            />
            <Button
              size="icon"
              className="gradient-bg text-primary-foreground rounded-xl shrink-0 border-0 hover:opacity-90 transition-opacity"
              onClick={handleSend}
              disabled={isLoading || (!input.trim() && !previewImage)}
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
          <p className="text-xs text-muted-foreground text-center mt-2">
            仅供参考，不构成医疗诊断建议
          </p>
        </div>
      </div>
    </div>
  );
};

export default Chat;
