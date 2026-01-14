import os
import sys
from search import search_prompt


def print_welcome():
    """Display welcome message"""
    print("=" * 60)
    print("  BEM-VINDO AO CHAT INTERATIVO")
    print("=" * 60)
    print("Digite suas perguntas e pressione Enter.")
    print("Digite 'sair' ou 'exit' para encerrar.\n")


def get_user_input():
    """Get question from user"""
    try:
        question = input("\n👤 Você: ").strip()
        return question
    except (KeyboardInterrupt, EOFError):
        print("\n\nEncerrando chat...")
        return None


def process_question(question):
    """Process question through RAG chain"""
    try:
        print("\n🤖 Assistente: ", end="", flush=True)
        response = search_prompt(question)
        print(response)
        return True
    except Exception as e:
        print(f"\n❌ Erro ao processar pergunta: {str(e)}")
        return False


def run_interactive_chat():
    """Run interactive chat loop"""
    print_welcome()
    
    while True:
        question = get_user_input()
        
        if question is None:
            break
            
        if question.lower() in ["sair", "exit", "quit", "q"]:
            print("\n👋 Até logo!")
            break
            
        if not question:
            print("⚠️  Por favor, digite uma pergunta.")
            continue
            
        process_question(question)


def main():
    """Main entry point for chat application"""
    try:
        print("🔧 Inicializando sistema...")
        print("✅ Sistema iniciado com sucesso!\n")
        run_interactive_chat()
        
    except KeyboardInterrupt:
        print("\n\n👋 Chat encerrado pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro fatal: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
