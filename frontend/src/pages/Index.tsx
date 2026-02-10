import { useRef } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { FileText, ScanLine, ClipboardList, ArrowRight, Sparkles, ChevronDown } from "lucide-react";
import { motion } from "framer-motion";

const features = [
  {
    icon: FileText,
    title: "病历解析",
    desc: "上传病历文档，智能解读诊断与用药方案，让医学术语变得通俗易懂。",
  },
  {
    icon: ScanLine,
    title: "影像分析",
    desc: "上传 X光、CT 等影像照片，获取专业且易理解的分析解读。",
  },
  {
    icon: ClipboardList,
    title: "疾病管理",
    desc: "长期跟踪病情变化，获取个性化的健康管理建议与提醒。",
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.15, duration: 0.6, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] },
  }),
};

const Index = () => {
  const introEndRef = useRef<HTMLDivElement | null>(null);
  const scrollToContent = () => introEndRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background */}
      <div className="fixed inset-0 -z-10">
        <div className="landing-backdrop" />
        <div className="landing-arc" />
        <div className="absolute top-[9%] left-[8%] w-[240px] h-[240px] rounded-full bg-primary/[0.08] blur-[85px]" />
        <div className="absolute bottom-[8%] right-[10%] w-[220px] h-[220px] rounded-full bg-accent/[0.08] blur-[85px]" />
      </div>

      {/* Nav */}
      <header className="glass-nav fixed top-0 left-0 right-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl gradient-bg flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="font-semibold text-lg tracking-tight">MedInsight</span>
          </div>
          <Link to="/chat">
            <Button className="gradient-bg text-primary-foreground rounded-xl px-6 hover:opacity-90 transition-opacity border-0">
              开始使用
            </Button>
          </Link>
        </div>
      </header>

      <main>
        {/* Intro */}
        <section className="min-h-screen pt-28 px-6 flex items-center">
          <div className="max-w-5xl mx-auto w-full text-center">
            <motion.p
              className="text-xs md:text-sm tracking-[0.24em] uppercase text-muted-foreground mb-6"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              Chronic Care Intelligence
            </motion.p>

            <motion.h1
              className="text-5xl md:text-7xl lg:text-8xl font-semibold tracking-tight leading-[1.02]"
              initial={{ opacity: 0, y: 28 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            >
              Care, Clearly.
              <br />
              <span className="gradient-text">Beyond The Clinic.</span>
            </motion.h1>

            <motion.p
              className="text-base md:text-lg text-muted-foreground max-w-2xl mx-auto mt-8"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.22 }}
            >
              Turn scattered reports into clear actions for long-term health.
            </motion.p>

            <motion.button
              type="button"
              onClick={scrollToContent}
              className="mt-12 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              Scroll to Explore
              <motion.span
                animate={{ y: [0, 6, 0] }}
                transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
              >
                <ChevronDown className="w-4 h-4" />
              </motion.span>
            </motion.button>
          </div>
        </section>

        <div ref={introEndRef} />

        {/* Hero */}
        <section className="pt-20 pb-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="inline-flex items-center gap-2 glass-card px-4 py-2 mb-8 text-sm text-muted-foreground">
              <Sparkles className="w-4 h-4 text-primary" />
              AI 驱动的医疗健康助手
            </div>
          </motion.div>

          <motion.h1
            className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.1] mb-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          >
            上传病历，
            <br />
            <span className="gradient-text">AI 为您解读</span>
          </motion.h1>

          <motion.p
            className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
          >
            让复杂的医学报告变得清晰易懂。上传病历或影像照片，获取专业的智能解析与个性化健康建议。
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
          >
            <Link to="/chat">
              <Button size="lg" className="gradient-bg text-primary-foreground rounded-2xl px-8 py-6 text-base hover:opacity-90 transition-opacity border-0 gap-2">
                开始智能解析
                <ArrowRight className="w-5 h-5" />
              </Button>
            </Link>
          </motion.div>
        </div>

        {/* Feature cards */}
        <div className="max-w-5xl mx-auto mt-24 grid md:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              className="glass-card-hover p-8 flex flex-col items-start gap-4"
              custom={i}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-50px" }}
              variants={fadeUp}
            >
              <div className="w-12 h-12 rounded-2xl gradient-bg-subtle flex items-center justify-center">
                <f.icon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">{f.title}</h3>
              <p className="text-muted-foreground leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* CTA */}
        <motion.div
          className="max-w-3xl mx-auto mt-24 glass-card p-10 text-center"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
        >
          <h2 className="text-2xl md:text-3xl font-bold mb-4">
            准备好了解您的健康状况了吗？
          </h2>
          <p className="text-muted-foreground mb-6">
            只需上传病历或影像照片，AI 将在几秒内为您提供专业解读。
          </p>
          <Link to="/chat">
            <Button size="lg" className="gradient-bg text-primary-foreground rounded-2xl px-8 hover:opacity-90 transition-opacity border-0 gap-2">
              立即体验
              <ArrowRight className="w-5 h-5" />
            </Button>
          </Link>
        </motion.div>
        </section>
      </main>

      {/* Footer */}
      <footer className="py-8 px-6 text-center text-sm text-muted-foreground">
        © 2026 MedInsight · 仅供参考，不构成医疗建议
      </footer>
    </div>
  );
};

export default Index;
