import { useRef } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { FileText, ScanLine, ClipboardList, ArrowRight, Sparkles, ChevronDown, Github } from "lucide-react";
import { motion } from "framer-motion";

const features = [
  {
    icon: FileText,
    title: "Medical Record Analysis",
    desc: "Upload records for clear diagnosis insights and medication guidance in plain language.",
  },
  {
    icon: ScanLine,
    title: "Image Analysis",
    desc: "Upload X-rays, CT scans, and other images for professional, easy-to-understand analysis.",
  },
  {
    icon: ClipboardList,
    title: "Health Management",
    desc: "Track your health over time with personalized health management tips and reminders.",
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
          <motion.div
            className="absolute top-[-28%] left-[-10%] w-[720px] h-[720px] rounded-full bg-primary/[0.24] blur-[220px]"
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.98, opacity: 0 }}
            transition={{ duration: 1.8, ease: [0.22, 1, 0.36, 1] }}
          />
          <motion.div
            className="absolute bottom-[-28%] right-[-10%] w-[700px] h-[700px] rounded-full bg-accent/[0.24] blur-[220px]"
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.98, opacity: 0 }}
            transition={{ duration: 1.8, ease: [0.22, 1, 0.36, 1], delay: 0.12 }}
          />
      </div>

      {/* Nav */}
      <header 
        className="fixed top-0 left-0 right-0 z-50 border-b-0"
        style={{
          background: 'linear-gradient(to bottom, hsla(230, 30%, 98%, 0.4) 0%, hsla(230, 30%, 98%, 0.1) 100%)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)'
        }}
      >
        <div className="w-full px-4 h-16 relative">
            <motion.div
            className="absolute left-14 top-0 h-16 flex items-center gap-2"
            initial={{ opacity: 0, x: -24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.45 }}
          >
            <div className="w-8 h-8 rounded-xl overflow-hidden flex items-center justify-center bg-white/0">
              <img src="/logo.png" alt="Medora" className="w-8 h-8 object-cover" onError={(e:any) => (e.target.src = '/placeholder.svg')} />
            </div>
            <span className="font-semibold text-lg tracking-tight" style={{ color: '#0e0b40ff'}}>Medora</span>
          </motion.div>

          <div className="absolute right-28 top-0 h-16 flex items-center">
            <div className="relative inline-block">
              <Link to="/chat">
                <Button 
                  className="rounded-full px-4 hover:shadow-md transition-all hover:scale-105"
                  style={{
                    background: 'rgba(99, 102, 241, 0.8)',
                    backdropFilter: 'blur(20px)',
                    border: '2px solid rgba(165,180,252, 0.5)',
                    boxShadow: '0 4px 16px rgba(99, 102, 241, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(99, 102, 241, 0.2)'
                  }}
                >
                  <span className="text-primary-foreground">Get started</span>
                </Button>
              </Link>

              <a
                href="https://github.com/wuli666/medgemma_afu"
                target="_blank"
                rel="noopener noreferrer"
                className="absolute top-1/2 -translate-y-1/2"
                style={{ left: 'calc(100% + 12px)' }}
              >
                <Button
                  className="w-10 h-10 p-0 rounded-full flex items-center justify-center transition-all hover:scale-105"
                  style={{
                    background: 'rgba(14,11,64,0.8)',
                    backdropFilter: 'blur(20px)',
                    WebkitBackdropFilter: 'blur(20px)',
                    border: '2px solid rgba(110, 106, 183, 0.4)',
                    boxShadow: '0 4px 12px rgba(14,11,64,0.45), inset 0 1px 0 rgba(255,255,255,0.4), inset 0 -1px 0 rgba(14,11,64,0.2)'
                  }}
                  aria-label="Open GitHub repository"
                >
                  <svg className="w-6 h-6 fill-current text-white" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" role="img" aria-hidden="false">
                    <title>GitHub</title>
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.385.6.11.82-.26.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.09-.745.083-.73.083-.73 1.205.085 1.84 1.238 1.84 1.238 1.07 1.835 2.807 1.305 3.492.998.108-.775.418-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.468-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23a11.5 11.5 0 013.003-.404c1.02.005 2.045.138 3.003.404 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.77.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.429.37.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .32.21.69.825.57C20.565 21.795 24 17.295 24 12 24 5.37 18.63 0 12 0z" />
                  </svg>
                </Button>
              </a>
            </div>
          </div>
        </div>
      </header>

      <main>
        {/* Intro */}
        <section className="min-h-screen pt-28 px-6 flex items-center">
          <div className="w-full text-center">
            <motion.p
              className=" -mt-20 text-xs md:text-sm tracking-[0.24em] uppercase text-muted-foreground mb-6"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              Chronic Care Intelligence
            </motion.p>

            <motion.h1
              className="text-5xl md:text-7xl lg:text-8xl font-semibold tracking-tight leading-[2]"
              // className="text-4xl md:text-6xl lg:text-7xl font-semibold tracking-tight leading-[2]"
              initial={{ opacity: 0, y: 28 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            >
{/* 
            <span style={{ color: '#0e0b40ff' }} className="block">
              {/* Care Clearly
              Medical Empathy */}
              {/* Medical Empathy
            </span> */}

            {/* <span className="gradient-text block mt-6 pb-4 text-4xl md:text-6xl lg:text-7xl font-semibold tracking-tight leading-[2]">
              {/* Beyond The Clinic. */}
              {/* Decoding Ongoing Records Always
            </span>
            </motion.h1> */}

            {/* <motion.p
              className="text-base md:text-lg text-muted-foreground max-w-2xl mx-auto mt-8"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.22 }}
            >
              {/* Turn scattered reports into clear actions for long-term health. */}
              {/* beyond The Clinic. */}
              {/* Care Between Visits
            </motion.p> */} 
            
            <div className="block">
              {"Medical Empathy".split(' ').map((w, i) => (
                <span key={i} className="inline-block mr-3">
                  <span className="inline-block font-extrabold text-[1.15em] md:text-[1.2em] lg:text-[1.25em] leading-[1]" style={{ color: '#0e0b40ff', opacity: 0.90 }}>
                    {w.charAt(0)}
                  </span>
                  <span className="inline-block" style={{ color: '#0e0b40ff', opacity: 0.80 }}>{w.slice(1)}</span>
                </span>
              ))}
            </div>

            <div className="block mt-6 pb-4 text-4xl md:text-6xl lg:text-7xl font-semibold tracking-tight lg:leading-[1.5]" style={{opacity: 0.90 }}>
              {"Decoding Ongoing Records Always".split(' ').map((w, i) => (
                <span key={i} className="inline-block mr-3">
                  <span className="inline-block font-extrabold gradient-text text-[1.15em] md:text-[1.2em] lg:text-[1.25em] leading-[1]">
                    {w.charAt(0)}
                  </span>
                  <span className="inline-block gradient-text opacity-80">{w.slice(1)}</span>
                </span>
              ))}
            </div>
            </motion.h1>
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
        <div className="max-w-4xl mx-auto text-center backdrop-opacity-30">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          >
            <div 
              className="inline-flex items-center gap-2 px-4 py-2 mb-8 text-sm rounded-full"
              style={{
                background: 'rgba(255, 255, 255, 0)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '2px solid rgba(165,180,252, 0.3)',
                boxShadow: '0 4px 16px rgba(99, 102, 241, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(99, 102, 241, 0.1)'
              }}
            >
              <Sparkles className="w-4 h-4 text-primary" />
              <span className="text-muted-foreground">AI-Powered Medical Health Assistant</span>
            </div>
          </motion.div>

          <motion.h1
            className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.1] mb-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          >
            <span style={{ color: '#0e0b40ff' }}>Upload medical records</span>
            <br />
            <span className="gradient-text">AI gets it</span>
          </motion.h1>

          <motion.p
            className="text-lg md:text-xl font-light mb-6"
            style={{ color: '#a8b3c4ff' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
          >
            Upload medical records or images for AI-powered analysis and personalized health insights.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
          >
          </motion.div>
        </div>
        {/* Feature cards */}
        <div className="max-w-6xl mx-auto mt-8 grid md:grid-cols-3 gap-10 backdrop-opacity-30">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              className="p-8 flex flex-col items-start gap-4 rounded-3xl"
              style={{
                background: 'rgba(255, 255, 255, 0)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '2px solid rgba(165,180,252, 0.3)',
                boxShadow: '0 4px 16px rgba(99, 102, 241, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 0 rgba(99, 102, 241, 0.1)'
              }}
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
          className="max-w-3xl mx-auto mt-12 flex flex-col md:flex-row items-center justify-center gap-6"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
        >
          <h2 className="text-xl md:text-2xl font-medium text-muted-foreground">
            Ready to discover your health status?
          </h2>
          
          <Link to="/chat">
            <Button 
              size="lg" 
              className="rounded-full px-8 gap-2 hover:shadow-md transition-all hover:scale-105"
              style={{
                background: 'rgba(99, 102, 241, 0.8)',
                backdropFilter: 'blur(20px)',
                border: '2px solid rgba(165,180,252, 0.8)',
                boxShadow: '0 4px 16px rgba(99, 102, 241, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.5), inset 0 -1px 0 rgba(99, 102, 241, 0.3)'
              }}
            >
              <span className="text-primary-foreground">Try Now</span>
              <ArrowRight className="w-5 h-5 text-primary-foreground" />
            </Button>
          </Link>
        </motion.div>
        </section>
      </main>

      {/* Footer */}
      <footer className="py-6 px-6 text-center text-sm text-muted-foreground">
        © 2026 Medora · For reference only
      </footer>
    </div>
  );
};

export default Index;
