import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { DottedSurface } from "@/components/DottedSurface";

export const metadata: Metadata = {
  title: "Auditr",
  description: "AI Financial Statement Auditor",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className="h-full antialiased"
    >
      <body className="min-h-full flex flex-col">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          <div className="fixed inset-0 z-0 bg-slate-50 dark:bg-[#121212]" />
          <DottedSurface className="pointer-events-none fixed inset-0 z-0 opacity-50 dark:opacity-30" />
          <div className="relative z-10 flex min-h-screen flex-col text-slate-900 dark:text-slate-100">
            {children}
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
