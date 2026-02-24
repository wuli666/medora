import { useState, useRef, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sparkles, Send, Paperclip, Plus, MessageSquare, X, Calendar, FileDown,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
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
  const isReportLike = /(^#\s)|(\n##\s)/m.test(show);

  return (
    <div
      className={`w-full max-w-none py-2 ${
        isReportLike
          ? "prose prose-sm prose-slate max-w-none rounded-2xl p-4 border border-white/50 bg-white/55 shadow-[0_8px_24px_rgba(148,163,184,0.18)]"
          : "prose prose-sm prose-neutral dark:prose-invert"
      }`}
      style={
        isReportLike
          ? {
              backdropFilter: "blur(14px)",
              WebkitBackdropFilter: "blur(14px)",
            }
          : undefined
      }
    >
      <ReactMarkdown
        components={{
          h1: ({ children }) => <h1 className="text-xl font-bold mb-3 text-slate-800">{children}</h1>,
          h2: ({ children }) => <h2 className="text-base font-semibold mt-4 mb-2 text-slate-700">{children}</h2>,
          p: ({ children }) => <p className="leading-7 text-slate-700 my-1">{children}</p>,
          ul: ({ children }) => <ul className="list-disc pl-5 my-1 space-y-1 text-slate-700">{children}</ul>,
          li: ({ children }) => <li className="leading-6">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-slate-800">{children}</strong>,
        }}
      >
        {show}
      </ReactMarkdown>
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
  substeps?: SubStepItem[];
  current_substep?: string;
};

type SubStepItem = {
  id: string;
  label: string;
  status: string;
  detail?: string;
};

type MultiAgentResponse = {
  run_id: string;
  summary: string;
  planner: string;
  tool: string;
  search: string;
  reflect_verify: string;
  stages: StageItem[];
  summary_struct?: Record<string, unknown>;
  report_download_url?: string;
};

type UiMessage = ChatMessage & {
  stages?: StageItem[];
  reportDownloadUrl?: string;
};
type LoadingStage = {
  key: string;
  label: string;
  status?: string;
  content?: string;
  substeps?: SubStepItem[];
  current_substep?: string;
};

type Lang = "zh" | "en";

// 日历事件类型
type CalendarEvent = {
  id: string;
  date: Date;
  type: "medication" | "appointment" | "followup";
  title: Record<Lang, string>;
  description: Record<Lang, string>;
};

// 模拟日历数据
const mockCalendarEvents: CalendarEvent[] = [
  {
    id: "1",
    date: new Date(2026, 1, 10),
    type: "medication",
    title: { zh: "服用降压药", en: "Take blood pressure medication" },
    description: { zh: "每天早上8点服用", en: "Take it every morning at 8:00 AM" },
  },
  {
    id: "2",
    date: new Date(2026, 1, 15),
    type: "appointment",
    title: { zh: "心内科复诊", en: "Cardiology follow-up visit" },
    description: { zh: "王医生专家门诊", en: "Specialist clinic with Dr. Wang" },
  },
  {
    id: "3",
    date: new Date(2026, 1, 20),
    type: "followup",
    title: { zh: "血糖复查", en: "Blood glucose recheck" },
    description: { zh: "空腹血糖检测", en: "Fasting blood glucose test" },
  },
  {
    id: "4",
    date: new Date(2026, 1, 25),
    type: "medication",
    title: { zh: "糖尿病用药调整", en: "Diabetes medication adjustment" },
    description: { zh: "二甲双胍增量", en: "Increase metformin dosage" },
  },
];

const formatTime = (date: Date, lang: Lang) =>
  new Intl.DateTimeFormat(lang === "zh" ? "zh-CN" : "en-US", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);

const STAGE_CLASS: Record<string, string> = {
  // use the former border colors as the background
  quick_router: "bg-indigo-200/60",
  tooler: "bg-sky-200/60",
  searcher: "bg-violet-200/60",
  planner: "bg-cyan-200/60",
  reflector: "bg-amber-200/60",
  summarize: "bg-emerald-200/60",
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 1200000;
const DEFAULT_LOADING_FLOW: LoadingStage[] = [
  { key: "quick_router", label: "意图识别", status: "running", content: "" },
];

const STAGE_LABELS: Record<string, { zh: string; en: string }> = {
  quick_router: { zh: "意图识别", en: "Intent recognition" },
  tooler: { zh: "病历/影像解析", en: "Record/Image parsing" },
  searcher: { zh: "医学检索补充", en: "Medical search" },
  planner: { zh: "管理计划生成", en: "Plan generation" },
  reflector: { zh: "一致性校验", en: "Consistency check" },
  summarize: { zh: "患者摘要", en: "Patient summary" },
};

const TEXT: Record<Lang, Record<string, string>> = {
  zh: {
    history: "咨询历史",
    calendar: "健康日历",
    newSession: "新会话",
    year: "年",
    month: "月",
    medication: "用药",
    appointment: "就诊",
    recheck: "复查",
    stageDetail: "查看阶段详情",
    running: "正在执行：",
    waitingStages: "等待后端阶段回传",
    noStageOutput: "无阶段输出",
    thinking: "思考中",
    stageSkipped: "已跳过该阶段",
    downloadReport: "下载报告 PDF",
    welcomeTitle: "今天需要 Medora 做点什么？",
    welcomeDesc: "我可以帮您解读病历、分析影像，并生成随访建议；也可以结合您的历史情况提供个性化的健康评估与管理方案。",
    quickTaskReport: "请用通俗语言解释这份检查报告",
    quickTaskFollowup: "帮我梳理复诊前需要关注的三个重点",
    quickTaskPlan: "根据目前情况给我一周的健康管理建议",
    placeholder: "输入您的问题，或上传病历/影像...",
    disclaimer: "仅供参考，不构成医疗诊断建议",
    defaultImagePrompt: "请分析这张影像",
    defaultPdfPrompt: "请分析这份PDF报告",
    noSummary: "未返回总结内容",
    requestFailed: "请求失败",
    requestTimeout: "请求超时",
    retryLater: "请稍后重试。",
    unknownError: "未知错误",
    uploadedImageAlt: "上传的图片",
    previewAlt: "预览",
  },
  en: {
    history: "History",
    calendar: "Health calendar",
    newSession: "New session",
    year: "Year",
    month: "Month",
    medication: "Medication",
    appointment: "Appointment",
    recheck: "Recheck",
    stageDetail: "View stages details",
    running: "Running:",
    waitingStages: "waiting for backend stages",
    noStageOutput: "No stage output",
    thinking: "Thinking...",
    stageSkipped: "Stage skipped",
    downloadReport: "Download report PDF",
    welcomeTitle: "What would you like Medora to help with today?",
    welcomeDesc: "I can explain medical records and images, generate follow-up suggestions, and provide personalized health management guidance.",
    quickTaskReport: "Explain this report in plain language",
    quickTaskFollowup: "Summarize three priorities before my follow-up visit",
    quickTaskPlan: "Create a one-week health plan from my current status",
    placeholder: "Type your question, or upload records/images...",
    disclaimer: "For reference only — not medical advice.",
    defaultImagePrompt: "Please analyze this image",
    defaultPdfPrompt: "Please analyze this PDF report",
    noSummary: "No summary returned",
    requestFailed: "Request failed",
    requestTimeout: "Request timed out",
    retryLater: "Please try again later.",
    unknownError: "Unknown error",
    uploadedImageAlt: "Uploaded image",
    previewAlt: "Preview",
  },
};

const getStageLabel = (key: string, label: string | undefined, lang: Lang) =>
  STAGE_LABELS[key]?.[lang] || label || key;

const buildStagesFromLegacy = (payload: Partial<MultiAgentResponse>, lang: Lang): StageItem[] => {
  if (payload.stages && payload.stages.length > 0) {
    return payload.stages.map((stage) => ({
      ...stage,
      label: getStageLabel(stage.key, stage.label, lang),
    }));
  }
  return [
    {
      key: "tooler",
      label: STAGE_LABELS.tooler[lang],
      status: payload.tool ? "done" : "skipped",
      content: payload.tool || "",
      substeps: [],
      current_substep: "",
    },
    {
      key: "searcher",
      label: STAGE_LABELS.searcher[lang],
      status: payload.search ? "done" : "skipped",
      content: payload.search || "",
      substeps: [],
      current_substep: "",
    },
    {
      key: "planner",
      label: STAGE_LABELS.planner[lang],
      status: payload.planner ? "done" : "skipped",
      content: payload.planner || "",
      substeps: [],
      current_substep: "",
    },
    {
      key: "reflector",
      label: STAGE_LABELS.reflector[lang],
      status: payload.reflect_verify ? "done" : "skipped",
      content: payload.reflect_verify || "",
      substeps: [],
      current_substep: "",
    },
    {
      key: "summarize",
      label: STAGE_LABELS.summarize[lang],
      status: payload.summary ? "done" : "skipped",
      content: payload.summary || "",
      substeps: [],
      current_substep: "",
    },
  ];
};

/* ── Main page ── */
const Chat = () => {
  const [sessions] = useState<ChatSession[]>(mockSessions);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [calendarOpen, setCalendarOpen] = useState(false);
  // preview 状态：页面初次进入时仅展示展开后的标题区域
  const [sidebarPreview, setSidebarPreview] = useState(false);
  const [calendarPreview, setCalendarPreview] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isCentered, setIsCentered] = useState(true);
  const [lang, setLang] = useState<Lang>("en");
  const [loadingFlow, setLoadingFlow] = useState<LoadingStage[]>([]);
  const [loadingStageIdx, setLoadingStageIdx] = useState(0);
  const [loadingVisibleCount, setLoadingVisibleCount] = useState(1);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const [selectedPdfFile, setSelectedPdfFile] = useState<File | null>(null);
  const [yearSelectorOpen, setYearSelectorOpen] = useState(false);
  const [monthSelectorOpen, setMonthSelectorOpen] = useState(false);
  const [displayYear, setDisplayYear] = useState<number>(new Date().getFullYear());
  const [displayMonth, setDisplayMonth] = useState<number>(new Date().getMonth());
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
    const flow = stages.map((s) => ({
      key: s.key,
      label: getStageLabel(s.key, s.label, lang),
      status: s.status,
      content: s.content,
      substeps: s.substeps || [],
      current_substep: s.current_substep || "",
    }));
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

  useEffect(() => {
    setLoadingFlow((prev) =>
      prev.map((stage) => ({
        ...stage,
        label: getStageLabel(stage.key, stage.label, lang),
      })),
    );
    setMessages((prev) =>
      prev.map((msg) => ({
        ...msg,
        stages: msg.stages?.map((stage) => ({
          ...stage,
          label: getStageLabel(stage.key, stage.label, lang),
        })),
      })),
    );
  }, [lang]);

  // 页面加载时临时以 "标题预览" 形式展开左侧面板，3 秒后恢复为未展开状态
  useEffect(() => {
    setSidebarPreview(true);
    setCalendarPreview(true);
    setSidebarOpen(true);
    setCalendarOpen(true);
    const t = setTimeout(() => {
      setSidebarOpen(false);
      setCalendarOpen(false);
      setSidebarPreview(false);
      setCalendarPreview(false);
    }, 3000);
    return () => clearTimeout(t);
  }, []);
  const toggleLang = () => setLang((s) => (s === "zh" ? "en" : "zh"));

  const handleSend = async () => {
    setIsCentered(false);
    const text = input.trim();
    if (!text && !previewImage && !selectedPdfFile) return;
    const outboundText = text || (selectedPdfFile ? TEXT[lang].defaultPdfPrompt : TEXT[lang].defaultImagePrompt);

    const userMsg: UiMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: outboundText,
      imageUrl: previewImage ?? undefined,
      timestamp: new Date(),
    };
    setMessages((m) => [...m, userMsg]);
    setIsLoading(true);
    setLoadingFlow(
      DEFAULT_LOADING_FLOW.map((stage) => ({
        ...stage,
        label: getStageLabel(stage.key, stage.label, lang),
      })),
    );
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
      if (selectedPdfFile) {
        formData.append("pdf", selectedPdfFile);
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
        throw new Error(payload.detail || `${TEXT[lang].requestFailed}: ${response.status}`);
      }

      const aiMsg: UiMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: payload.summary || TEXT[lang].noSummary,
        timestamp: new Date(),
        stages: buildStagesFromLegacy(payload, lang),
        reportDownloadUrl: payload.report_download_url || undefined,
      };
      setMessages((m) => [...m, aiMsg]);
      setInput("");
      setPreviewImage(null);
      setSelectedImageFile(null);
      setSelectedPdfFile(null);
      if (fileRef.current) fileRef.current.value = "";
    } catch (error) {
      const aiMsg: UiMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          error instanceof DOMException && error.name === "AbortError"
            ? `${TEXT[lang].requestTimeout} (> ${REQUEST_TIMEOUT_MS / 1000}s), ${TEXT[lang].retryLater}`
            : `${TEXT[lang].requestFailed}: ${error instanceof Error ? error.message : TEXT[lang].unknownError}`,
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
    if (file.type === "application/pdf") {
      setSelectedPdfFile(file);
      setSelectedImageFile(null);
      setPreviewImage(null);
    } else {
      setSelectedImageFile(file);
      setSelectedPdfFile(null);
      const reader = new FileReader();
      reader.onload = () => setPreviewImage(reader.result as string);
      reader.readAsDataURL(file);
    }
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

  const handleDownloadReport = (reportPath: string) => {
    if (!reportPath) return;
    const url = `${API_BASE}${reportPath}`;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const EMPTY_TASKS = [
    {
      text: TEXT[lang].quickTaskReport,
      hasIcon: true,
      onClick: () => {
        setInput(TEXT[lang].quickTaskReport);
        fileRef.current?.click();
      },
    },
    {
      text: TEXT[lang].quickTaskFollowup,
      hasIcon: false,
      onClick: () => handleQuickPrompt(TEXT[lang].quickTaskFollowup),
    },
    {
      text: TEXT[lang].quickTaskPlan,
      hasIcon: false,
      onClick: () => handleQuickPrompt(TEXT[lang].quickTaskPlan),
    },
  ];

  return (
    <div className="h-screen flex flex-col relative overflow-hidden">
      <style>{`
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
        .no-scrollbar::-webkit-scrollbar { display: none; }
      `}</style>
      {/* Background */}
      <div className="fixed inset-0 -z-10">
        <motion.div
          className="absolute top-[-30%] left-[-15%] w-[760px] h-[760px] rounded-full bg-primary/[0.24] blur-[220px]"
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.98, opacity: 0 }}
          transition={{ duration: 1.8, ease: [0.22, 1, 0.36, 1] }}
        />
        <motion.div
          className="absolute bottom-[-30%] right-[-15%] w-[740px] h-[740px] rounded-full bg-accent/[0.24] blur-[220px]"
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.98, opacity: 0 }}
          transition={{ duration: 1.8, ease: [0.22, 1, 0.36, 1], delay: 0.12 }}
        />
      </div>

      {/* Main layout */}
      <div className="flex flex-col h-full">
        {/* Top bar */}
        <header className="px-4 h-16 flex items-center shrink-0 flex-shrink-0">
          <Link to="/">
            <motion.div
              className="flex items-center gap-2"
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.45 }}
            >
              <div className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center bg-white/0">
                <img src="/logo.png" alt="Medora" className="w-8 h-8 object-cover" onError={(e:any) => (e.target.src = '/placeholder.svg')} />
              </div>
              <span className="font-semibold text-lg" style={{ color: '#0e0b40ff'}}>Medora</span>
            </motion.div>
          </Link>
          <div className="absolute right-4 top-0 h-16 flex items-center">
            <button
              type="button"
              onClick={toggleLang}
              className="absolute top-1/2 -translate-y-1/2"
              style={{ left: "calc(100% - 48px)" }}
              aria-label="Toggle language"
            >
              <Button
                className="w-10 h-10 p-0 rounded-full flex items-center justify-center transition-all hover:scale-105"
                style={{
                  background: "rgba(14,11,64,0.8)",
                  backdropFilter: "blur(20px)",
                  WebkitBackdropFilter: "blur(20px)",
                  border: "2px solid rgba(110, 106, 183, 0.4)",
                  boxShadow: "0 4px 12px rgba(14,11,64,0.45), inset 0 1px 0 rgba(255,255,255,0.4), inset 0 -1px 0 rgba(14,11,64,0.2)",
                }}
              >
                <span className="text-white font-medium">{lang === "zh" ? "EN" : "中"}</span>
              </Button>
            </button>
          </div>
        </header>

        {/* Main content with sidebar buttons */}
        <div className="flex-1 flex min-h-0 relative">
          {/* Left sidebar buttons - iOS 26 glass style */}
          {!sidebarOpen && (
            <div className="pl-2 pr-2 py-2 flex flex-col gap-2 absolute top-0 left-0">
              <button
                onClick={() => {
                  setSidebarOpen((prev) => {
                    const next = !prev;
                    if (next) setCalendarOpen(false);
                    return next;
                  });
                }}
                className="w-12 h-12 rounded-full flex items-center justify-center transition-all backdrop-blur-md bg-white/30 text-muted-foreground hover:text-foreground hover:bg-white/50 border border-white/20 shadow-sm"
                title={TEXT[lang].history}
                style={{
                  boxShadow: '0 2px 8px rgba(255,255,255,0.3), inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -1px 0 rgba(255,255,255,0.2)'
                }}
              >
                <MessageSquare className="w-5 h-5" />
              </button>
              {!calendarOpen && (
                <button
                  onClick={() => {
                    setCalendarOpen(true);
                    setSidebarOpen(false);
                  }}
                  className="w-12 h-12 rounded-full flex items-center justify-center transition-all backdrop-blur-md bg-white/30 text-muted-foreground hover:text-foreground hover:bg-white/50 border border-white/20 shadow-sm"
                  title={TEXT[lang].calendar}
                  style={{
                    boxShadow: '0 2px 8px rgba(255,255,255,0.3), inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -1px 0 rgba(255,255,255,0.2)'
                  }}
                >
                  <Calendar className="w-5 h-5" />
                </button>
              )}
            </div>
          )}

          {/* History sidebar - expanded from icon position */}
          <AnimatePresence>
            {sidebarOpen && (
              <motion.div
                className="absolute"
                style={{ left: 8, top: 8 }}
                initial={{ width: 48, height: 48, opacity: 0 }}
                animate={sidebarPreview ? { width: 280, height: 48, opacity: 1 } : { width: 280, height: 'calc(100% - 16px)', opacity: 1 }}
                exit={{ width: 48, height: 48, opacity: 0 }}
                transition={{ type: "spring", damping: 25, stiffness: 300 }}
              >
                <div 
                  className={sidebarPreview ? "rounded-2xl overflow-hidden flex" : "h-full rounded-2xl overflow-hidden flex flex-col"}
                  style={{
                    background: 'rgba(255, 255, 255, 0.35)',
                    backdropFilter: 'blur(20px)',
                    border: '2px solid rgba(255, 255, 255, 0.4)',
                    boxShadow: '0 8px 32px rgba(255, 255, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6), inset 0 -1px 0 rgba(255, 255, 255, 0.2)'
                  }}
                >
                  {/* Header */}
                  <div className="flex items-center justify-between p-3">
                    <button 
                      onClick={() => setSidebarOpen(false)}
                      className="flex items-center gap-2 text-foreground hover:opacity-80 transition-opacity"
                    >
                      <MessageSquare className="w-5 h-5" />
                      <span className="font-medium text-sm">{TEXT[lang].history}</span>
                    </button>
                  </div>
                  {/* Content: preview 时不渲染 */}
                  {!sidebarPreview && (
                    <div className="flex-1 overflow-y-auto p-2 space-y-1">
                      <Button variant="ghost" className="w-full justify-start gap-3 mb-2 rounded-xl text-sm h-10 hover:bg-white/50 text-foreground hover:text-foreground">
                        <Plus className="w-5 h-5" /> {TEXT[lang].newSession}
                      </Button>
                      {sessions.map((s) => (
                        <button
                          key={s.id}
                          className="w-full text-left p-3 rounded-xl hover:bg-white/30 transition-colors mb-1"
                        >
                          <div className="min-w-0">
                            <p className="text-sm font-medium truncate">{s.title}</p>
                            <p className="text-xs text-muted-foreground truncate">{s.lastMessage}</p>
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Calendar sidebar - expanded below history sidebar */}
          <AnimatePresence>
            {calendarOpen && (
              <motion.div
                className="absolute"
                style={{ left: 8, top: 64 }}
                initial={{ width: 48, height: 48, opacity: 0 }}
                animate={calendarPreview ? { width: 280, height: 48, opacity: 1 } : { width: 280, height: 394, opacity: 1 }}
                exit={{ width: 48, height: 48, opacity: 0 }}
                transition={{ type: "spring", damping: 25, stiffness: 300 }}
              >
                <div 
                  className={calendarPreview ? "rounded-2xl overflow-hidden flex" : "h-full rounded-2xl overflow-hidden flex flex-col"}
                  style={{
                    background: 'rgba(255, 255, 255, 0.35)',
                    backdropFilter: 'blur(20px)',
                    border: '2px solid rgba(255, 255, 255, 0.4)',
                    boxShadow: '0 8px 32px rgba(255, 255, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6), inset 0 -1px 0 rgba(255, 255, 255, 0.2)'
                  }}
                >
                  {/* Header */}
                  <div className="flex items-center justify-between p-3 pl-2">
                    <button 
                      onClick={() => setCalendarOpen(false)}
                      className="flex items-center gap-2 text-foreground hover:opacity-80 transition-opacity"
                    >
                      <Calendar className="w-5 h-5" />
                      <span className="font-medium text-sm">{TEXT[lang].calendar}</span>
                    </button>
                  </div>
                  {/* Content: preview 时不渲染 */}
                  {!calendarPreview && (
                    <div className="flex-1 flex flex-col p-2">
                      <div className="mb-2">
                        <div className="relative inline-flex items-center justify-center gap-1">
                          <div className="relative">
                            <button
                              type="button"
                              onClick={() => { setYearSelectorOpen((s) => !s); setMonthSelectorOpen(false); }}
                              className="px-2 py-1 rounded-xl bg-transparent hover:bg-white/5 transition-colors focus:outline-none ml-[4px]"
                            >
                              <span className="text-sm font-medium">{displayYear}</span>
                            </button>
                            {yearSelectorOpen && (
                                <div className="absolute left-0 top-full mt-2 z-50 w-24 max-h-40 overflow-y-auto no-scrollbar p-1 rounded-xl bg-white shadow-lg">
                                  {Array.from({ length: 11 }).map((_, i) => {
                                    const y = displayYear - 5 + i;
                                    const isFirst = i === 0;
                                    const isLast = i === 10;
                                    return (
                                      <button
                                        key={y}
                                        type="button"
                                        onClick={() => { setDisplayYear(y); setYearSelectorOpen(false); }}
                                        className={`w-full text-center py-2 hover:bg-primary/10 ${y === displayYear ? 'font-semibold' : ''} ${isFirst ? 'rounded-t-xl' : ''} ${isLast ? 'rounded-b-xl' : ''}`}
                                      >
                                        {y}
                                      </button>
                                    );
                                  })}
                                </div>
                              )}
                          </div>

                          <span className="text-sm">{TEXT[lang].year}</span>

                          <div className="relative">
                            <button
                              type="button"
                              onClick={() => { setMonthSelectorOpen((s) => !s); setYearSelectorOpen(false); }}
                              className="px-2 py-1 rounded-xl bg-transparent hover:bg-white/5 transition-colors focus:outline-none ml-[2px]"
                            >
                              <span className="text-sm font-medium">{displayMonth + 1}</span>
                            </button>
                            {monthSelectorOpen && (
                                <div className="absolute left-0 top-full mt-2 z-50 w-20 max-h-40 overflow-y-auto no-scrollbar p-1 rounded-xl bg-white shadow-lg">
                                  {Array.from({ length: 12 }).map((_, i) => {
                                    const m = i;
                                    const isFirst = i === 0;
                                    const isLast = i === 11;
                                    return (
                                      <button
                                        key={m}
                                        type="button"
                                        onClick={() => { setDisplayMonth(m); setMonthSelectorOpen(false); }}
                                        className={`w-full text-center py-2 hover:bg-primary/10 ${m === displayMonth ? 'font-semibold' : ''} ${isFirst ? 'rounded-t-xl' : ''} ${isLast ? 'rounded-b-xl' : ''}`}
                                      >
                                        {m + 1}
                                      </button>
                                    );
                                  })}
                                </div>
                              )}
                          </div>
                          <span className="text-sm">{TEXT[lang].month}</span>
                        </div>
                      </div>

                      <div className="overflow-y-auto no-scrollbar space-y-2" style={{ maxHeight: '252px' }}>
                        {mockCalendarEvents.map((event) => (
                          <div
                            key={event.id}
                            className={`p-3 rounded-xl border-l-3 ${
                              event.type === "medication"
                                ? "border-blue-500 bg-blue-50/60"
                                : event.type === "appointment"
                                  ? "border-red-500 bg-red-50/60"
                                  : "border-amber-500 bg-amber-50/60"
                            }`}
                          >
                            <div className="flex items-start justify-between">
                              <div>
                                <p className="text-sm font-medium">{event.title[lang]}</p>
                                <p className="text-xs text-muted-foreground mt-0.5">{event.description[lang]}</p>
                              </div>
                              <span className="text-xs text-muted-foreground">
                                {event.date.getMonth() + 1}/{event.date.getDate()}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Legend */}
                      <div className="mt-4 border-t border-white/20">
                        <div className="flex items-center justify-start w-full gap-4 pl-3 text-xs text-muted-foreground">
                          <div className="flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full bg-blue-500" />
                            <span>{TEXT[lang].medication}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full bg-red-500" />
                            <span>{TEXT[lang].appointment}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full bg-amber-500" />
                            <span>{TEXT[lang].recheck}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Messages */}
        <div ref={scrollRef} className={`flex-1 overflow-y-auto px-4 py-6 ${isCentered ? "flex" : ""}`}>
          <div
            className={isCentered ? "max-w-4xl mx-auto flex-1 flex flex-col justify-center" : "max-w-4xl mx-auto flex flex-col min-h-full"}
            style={isCentered ? { transform: "translateY(-6vh)" } : undefined}
          >
          {messages.map((msg, idx) => {
            const isSingleLine =
              typeof msg.content === "string" && !msg.content.includes("\n") && msg.content.length <= 60;
            return (
              <motion.div
                key={msg.id}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} mb-4`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
              >
                {msg.role === "user" ? (
                  <div className="max-w-[85%] md:max-w-[70%] space-y-2">
                    {msg.imageUrl && (
                      <div className="rounded-2xl overflow-hidden border border-border/50 glass-card">
                        <img src={msg.imageUrl} alt={TEXT[lang].uploadedImageAlt} className="max-h-64 object-contain w-full" />
                      </div>
                    )}
                    <div
                      className="gradient-bg text-primary-foreground p-4 rounded-2xl rounded-br-md"
                      style={{
                        background: 'rgba(99, 102, 241, 0.8)',
                        backdropFilter: 'blur(20px)',
                        WebkitBackdropFilter: 'blur(20px)',
                        // border: '2px solid rgba(165,180,252, 0.8)',
                        // boxShadow: '0 4px 16px rgba(99, 102, 241, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(99, 102, 241, 0.2)',
                        color: 'rgba(255, 255, 255, 1)',
                        borderRadius: isSingleLine ? '9999px' : '20px',
                        whiteSpace: isSingleLine ? 'nowrap' : 'normal',
                        overflow: isSingleLine ? 'hidden' : 'visible',
                        textOverflow: isSingleLine ? 'ellipsis' : 'clip',
                        height: isSingleLine ? '46px' : undefined,
                        display: isSingleLine ? 'flex' : undefined,
                        alignItems: isSingleLine ? 'center' : undefined,
                        padding: isSingleLine ? '0 16px' : undefined
                      }}
                    >
                      <p className="text-sm leading-relaxed ">{msg.content}</p>
                    </div>
                    <p className="text-[11px] text-muted-foreground text-right pr-1">
                      {formatTime(msg.timestamp, lang)}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {msg.reportDownloadUrl && (
                      <div className="flex justify-start mb-1">
                        <Button
                          type="button"
                          onClick={() => handleDownloadReport(msg.reportDownloadUrl!)}
                          className="rounded-xl h-9 px-4 text-sm text-white"
                          style={{
                            background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.88), rgba(14, 116, 144, 0.88))',
                            backdropFilter: 'blur(20px)',
                            WebkitBackdropFilter: 'blur(20px)',
                            border: '2px solid rgba(209, 250, 229, 0.65)',
                            boxShadow: '0 4px 12px rgba(16, 185, 129, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.3)',
                          }}
                        >
                          <FileDown className="w-4 h-4 mr-2" />
                          {TEXT[lang].downloadReport}
                        </Button>
                      </div>
                    )}
                    <AssistantBubble
                      content={msg.content}
                      isLatest={idx === messages.length - 1}
                    />
                    {msg.stages && msg.stages.length > 0 && (
                      <details className="pl-1">
                        <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground transition-colors">
                          {TEXT[lang].stageDetail}
                        </summary>
                        <div className="space-y-2 mt-2">
                          {msg.stages.map((stage) => {
                            const isCurrentStage = stage.status === "running";
                            return (
                              <div
                                key={`${msg.id}-${stage.key}`}
                                className={`rounded-xl p-3 ${STAGE_CLASS[stage.key] || "bg-muted/50"}`}
                                style={{
                                  backdropFilter: 'blur(20px)',
                                  WebkitBackdropFilter: 'blur(20px)',
                                  background: 'rgba(255, 255, 255, 0.6)',
                                  boxShadow: '0 8px 24px rgba(255, 255, 255, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.36)',
                                  opacity: isCurrentStage ? 1 : 0.5,
                                }}
                              >
                                <p className="text-xs font-semibold" style={{ color: '#0e0b40ff' }}>{stage.label}</p>
                                <p className="text-xs mt-1 max-h-24 overflow-auto" style={{ color: '#0e0b40ff' }}>{stage.content || TEXT[lang].noStageOutput}</p>
                                {stage.substeps && stage.substeps.length > 0 && (
                                  <div className="mt-2 space-y-1">
                                    {stage.substeps.map((sub) => {
                                      const isCurrent = stage.current_substep === sub.id && sub.status === "running";
                                      const dotClass = sub.status === "done"
                                        ? "bg-emerald-500"
                                        : sub.status === "error"
                                          ? "bg-red-500"
                                          : isCurrent
                                            ? "bg-primary"
                                            : "bg-muted-foreground/50";
                                      return (
                                        <div key={`${msg.id}-${stage.key}-${sub.id}`} className="flex items-start gap-2">
                                          <span className={`mt-1 h-1.5 w-1.5 rounded-full ${dotClass}`} />
                                          <div className="text-[11px] leading-4" style={{ color: '#0e0b40ff' }}>
                                            <div className={isCurrent ? "font-semibold" : ""}>{sub.label}</div>
                                            {sub.detail && <div className="opacity-80">{sub.detail}</div>}
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </details>
                    )}
                  </div>
                )}
              </motion.div>
            );
          })}

          {messages.length === 0 && (
            <motion.div
              className={isCentered ? "flex justify-center items-center mt-0 h-full" : "flex justify-center mt-auto"}
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="p-6 rounded-3xl max-w-4xl mx-auto w-full">
                <div className="flex flex-col items-center">
                  <div className="mt-4 text-3xl font-extrabold text-foreground/90 leading-relaxed" style={{ color: "#0e0b40ff" }}>
                    {TEXT[lang].welcomeTitle}
                  </div>
                  <div className="mt-2 text-muted-foreground leading-relaxed text-center max-w-2xl">
                    {TEXT[lang].welcomeDesc}
                  </div>
                  <div className="mt-5 grid gap-3 sm:grid-cols-3 w-full">
                    {EMPTY_TASKS.map((task) => (
                      <button
                        key={task.text}
                        type="button"
                        className="p-4 rounded-2xl border border-border/60 bg-white/60 hover:bg-white/80 transition-colors flex items-center justify-center text-center"
                        onClick={task.onClick}
                      >
                        <span className="text-sm">{task.text}</span>
                      </button>
                    ))}
                  </div>

                  {isCentered && (
                    <div className="mt-6 w-full px-4">
                      <div className="flex items-end gap-2 max-w-4xl mx-auto">
                        <div className="flex justify-center h-[46px]">
                          <Button
                            size="icon"
                            className="rounded-full shrink-0 border-0 transition-all hover:shadow-md w-9 h-9"
                            style={{
                              background: "rgba(255, 255, 255, 0.8)",
                              backdropFilter: "blur(20px)",
                              border: "2px solid rgba(255, 255, 255, 0.9)",
                              boxShadow: "0 4px 16px rgba(255, 255, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6), inset 0 -1px 0 rgba(255, 255, 255, 0.2)",
                            }}
                            onClick={() => fileRef.current?.click()}
                          >
                            <Paperclip className="w-4 h-4" style={{ color: "rgb(99, 102, 241)" }} />
                          </Button>
                        </div>
                        <div
                          className="flex-1 p-2 rounded-[36px] min-w-0"
                          style={{
                            background: "rgba(255, 255, 255, 0.35)",
                            backdropFilter: "blur(20px)",
                            border: "2px solid rgba(255, 255, 255, 0.4)",
                            boxShadow: "0 8px 32px rgba(255, 255, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6), inset 0 -1px 0 rgba(255, 255, 255, 0.2)",
                          }}
                        >
                          <input type="file" ref={fileRef} className="hidden" accept="image/*,application/pdf" onChange={handleFile} />
                          <Textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={TEXT[lang].placeholder}
                            className="w-full min-h-[36px] max-h-24 resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 text-sm text-foreground placeholder:text-muted-foreground py-2"
                            style={{ fieldSizing: "content" as any }}
                          />
                        </div>
                        <div className="flex justify-center h-[46px]">
                          <Button
                            size="icon"
                            className="rounded-full shrink-0 border-0 transition-all hover:shadow-md w-9 h-9"
                            style={{
                              background: "rgba(99, 102, 241, 0.8)",
                              backdropFilter: "blur(20px)",
                              border: "2px solid rgba(165,180,252, 0.8)",
                              boxShadow: "0 4px 16px rgba(99, 102, 241, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(99, 102, 241, 0.2)",
                            }}
                            onClick={handleSend}
                            disabled={isLoading || (!input.trim() && !previewImage && !selectedPdfFile)}
                          >
                            <Send className="w-4 h-4 text-primary-foreground" />
                          </Button>
                        </div>
                      </div>
                      {previewImage && (
                        <AnimatePresence>
                          <motion.div
                            className="max-w-4xl mx-auto px-4 w-full mt-3"
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                          >
                            <div
                              className="p-2 pb-0 relative inline-flex items-center gap-2 rounded-xl group"
                              style={{
                                backdropFilter: "blur(20px)",
                                WebkitBackdropFilter: "blur(20px)",
                                justifyContent: "flex-start",
                              }}
                            >
                              <img src={previewImage} alt={TEXT[lang].previewAlt} className="h-20 object-cover rounded-sm" />
                              <Button
                                variant="ghost"
                                size="icon"
                                className="rounded-lg h-8 w-8 absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity bg-white/50 hover:bg-white transition-colors"
                                onClick={() => {
                                  setPreviewImage(null);
                                  setSelectedImageFile(null);
                                  if (fileRef.current) fileRef.current.value = "";
                                }}
                              >
                                <X className="w-4 h-4 text-black" />
                              </Button>
                            </div>
                          </motion.div>
                        </AnimatePresence>
                      )}
                      {selectedPdfFile && (
                        <AnimatePresence>
                          <motion.div
                            className="max-w-4xl mx-auto px-4 w-full mt-3"
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                          >
                            <div
                              className="p-2 pb-0 relative inline-flex items-center gap-2 rounded-xl group"
                              style={{
                                backdropFilter: "blur(20px)",
                                WebkitBackdropFilter: "blur(20px)",
                                justifyContent: "flex-start",
                              }}
                            >
                              <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/40">
                                <svg className="w-5 h-5 text-red-500 shrink-0" viewBox="0 0 24 24" fill="currentColor">
                                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zM14 3.5L18.5 8H14V3.5zM6 20V4h7v5h5v11H6z" />
                                  <text x="7" y="17" fontSize="6" fontWeight="bold" fill="currentColor">PDF</text>
                                </svg>
                                <span className="text-sm text-foreground truncate max-w-[200px]">{selectedPdfFile.name}</span>
                              </div>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="rounded-lg h-8 w-8 absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity bg-white/50 hover:bg-white transition-colors"
                                onClick={() => {
                                  setSelectedPdfFile(null);
                                  if (fileRef.current) fileRef.current.value = "";
                                }}
                              >
                                <X className="w-4 h-4 text-black" />
                              </Button>
                            </div>
                          </motion.div>
                        </AnimatePresence>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {isLoading && (
            <motion.div className="flex justify-start" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div
                className="p-4 min-w-[340px] rounded-2xl"
                style={{
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)'
                }}
              >
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
                    {TEXT[lang].running} {loadingFlow[loadingStageIdx]?.label || TEXT[lang].waitingStages}
                  </span>
                </div>
                <div className="mt-4 space-y-2">
                  {loadingFlow.slice(0, loadingVisibleCount).map((stage, idx) => {
                    const isDone = idx < loadingStageIdx || ["done", "skipped", "error"].includes(stage.status || "");
                    const isCurrent = idx === loadingStageIdx;
                    const stageText = (stage.content || "").trim();
                    const displayContent = stageText
                      || (isCurrent ? TEXT[lang].thinking : (stage.status === "skipped" ? TEXT[lang].stageSkipped : ""));
                    const substeps = stage.substeps || [];
                    const recentSubsteps = substeps.slice(Math.max(0, substeps.length - 4));
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
                            style={{ opacity: isCurrent ? 1 : 0.5 }}
                          />
                          {idx !== loadingVisibleCount - 1 && (
                            <span
                              className={[
                                "mt-1 w-px h-6",
                                isDone ? "bg-emerald-400/80" : "bg-border/70",
                              ].join(" ")}
                            />
                          )}
                        </div>
                        <div
                          className={`rounded-xl p-3 w-full ${STAGE_CLASS[stage.key] || "bg-muted/50"}`}
                          style={{
                            backdropFilter: 'blur(20px)',
                            WebkitBackdropFilter: 'blur(20px)',
                            background: 'rgba(255, 255, 255, 0.6)',
                            boxShadow: '0 8px 24px rgba(255, 255, 255, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.36)',
                            opacity: isCurrent ? 1 : 0.6,
                          }}
                        >
                          <p className="text-xs font-semibold" style={{ color: '#0e0b40ff' }}>
                            {stage.label}
                          </p>
                          {displayContent && (
                            <p className="text-xs mt-1 max-h-28 overflow-auto whitespace-pre-wrap break-words" style={{ color: '#0e0b40ff' }}>
                              {displayContent}
                            </p>
                          )}
                          {recentSubsteps.length > 0 && (
                            <div className="mt-2 space-y-1">
                              {recentSubsteps.map((sub) => {
                                const isSubCurrent = stage.current_substep === sub.id && sub.status === "running";
                                const dotClass = sub.status === "done"
                                  ? "bg-emerald-500"
                                  : sub.status === "error"
                                    ? "bg-red-500"
                                    : isSubCurrent
                                      ? "bg-primary"
                                      : "bg-muted-foreground/50";
                                return (
                                  <div key={`${stage.key}-${sub.id}`} className="flex items-start gap-2">
                                    <span className={`mt-1 h-1.5 w-1.5 rounded-full ${dotClass}`} />
                                    <div className="text-[11px] leading-4" style={{ color: '#0e0b40ff' }}>
                                      <div className={isSubCurrent ? "font-semibold" : ""}>{sub.label}</div>
                                      {sub.detail && <div className="opacity-80">{sub.detail}</div>}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
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

        {/* Image preview (aligned with input, delete button shows on hover) */}
        <AnimatePresence>
          {previewImage && !isCentered && (
            <motion.div
              className="max-w-4xl mx-auto px-4 w-full "
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
            >
            <div
              className="p-2 pb-0 relative inline-flex items-center gap-2 rounded-xl group ml-[44px]"
              style={{
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
              }}
            >
                <img src={previewImage} alt={TEXT[lang].previewAlt} className="h-20 object-cover rounded-sm" />
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-lg h-8 w-8 absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity bg-white/50 hover:bg-white transition-colors"
                  onClick={() => {
                    setPreviewImage(null);
                    setSelectedImageFile(null);
                    if (fileRef.current) fileRef.current.value = "";
                  }}
                >
                  <X className="w-4 h-4 text-black" />
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* PDF preview */}
        <AnimatePresence>
          {selectedPdfFile && !isCentered && (
            <motion.div
              className="max-w-4xl mx-auto px-4 w-full"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
            >
              <div
                className="p-2 pb-0 relative inline-flex items-center gap-2 rounded-xl group ml-[44px]"
                style={{
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)',
                }}
              >
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/40">
                  <svg className="w-5 h-5 text-red-500 shrink-0" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zM14 3.5L18.5 8H14V3.5zM6 20V4h7v5h5v11H6z"/>
                    <text x="7" y="17" fontSize="6" fontWeight="bold" fill="currentColor">PDF</text>
                  </svg>
                  <span className="text-sm text-foreground truncate max-w-[200px]">{selectedPdfFile.name}</span>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-lg h-8 w-8 absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity bg-white/50 hover:bg-white transition-colors"
                  onClick={() => {
                    setSelectedPdfFile(null);
                    if (fileRef.current) fileRef.current.value = "";
                  }}
                >
                  <X className="w-4 h-4 text-black" />
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input */}
        {!isCentered && (
          <div className="p-4 shrink-0">
          <div className="flex items-end gap-2 max-w-4xl mx-auto">
            <div className="flex justify-center h-[46px]">
              <Button
                size="icon"
                className="rounded-full shrink-0 border-0 transition-all hover:shadow-md w-9 h-9"
                style={{
                  background: 'rgba(255, 255, 255, 0.8)',
                  backdropFilter: 'blur(20px)',
                  border: '2px solid rgba(255, 255, 255, 0.9)',
                  boxShadow: '0 4px 16px rgba(255, 255, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6), inset 0 -1px 0 rgba(255, 255, 255, 0.2)'
                }}
                onClick={() => fileRef.current?.click()}
              >
                <Paperclip className="w-4 h-4" style={{ color: 'rgb(99, 102, 241)' }} />
              </Button>
            </div>
            <div 
              className="flex-1 p-2 rounded-[36px] min-w-0"
              style={{
                background: 'rgba(255, 255, 255, 0.35)',
                backdropFilter: 'blur(20px)',
                border: '2px solid rgba(255, 255, 255, 0.4)',
                boxShadow: '0 8px 32px rgba(255, 255, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6), inset 0 -1px 0 rgba(255, 255, 255, 0.2)'
              }}
            >
              <input type="file" ref={fileRef} className="hidden" accept="image/*,application/pdf" onChange={handleFile} />
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={TEXT[lang].placeholder}
                className="w-full min-h-[36px] max-h-24 resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 text-sm text-foreground placeholder:text-muted-foreground py-2"
                style={{fieldSizing: 'content' as any }}
              />
            </div>
            <div className="flex justify-center h-[46px]">
              <Button
                size="icon"
                className="rounded-full shrink-0 border-0 transition-all hover:shadow-md w-9 h-9"
                style={{
                  background: 'rgba(99, 102, 241, 0.8)',
                  backdropFilter: 'blur(20px)',
                  border: '2px solid rgba(165,180,252, 0.8)',
                  boxShadow: '0 4px 16px rgba(99, 102, 241, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(99, 102, 241, 0.2)'
                }}
                onClick={handleSend}
                disabled={isLoading || (!input.trim() && !previewImage && !selectedPdfFile)}
              >
                <Send className="w-4 h-4 text-primary-foreground" />
              </Button>
            </div>
          </div>
          <p className="text-xs text-muted-foreground text-center mt-2">
            {TEXT[lang].disclaimer}
          </p>
        </div>
        )}
        {isCentered && (
          <p className="text-xs text-muted-foreground text-center" style={{ position: "fixed", left: 0, right: 0, bottom: 12 }}>
            {TEXT[lang].disclaimer}
          </p>
        )}
      </div>
    </div>
    {/* Close flex-col main layout */}
    </div>
    {/* Close outer container */}
    </div>
  );
};

export default Chat;
