import os

from src.controllers import ChatController


controller = ChatController()


def home_view() -> None:
    print("Bem vindo ao ChatPDF!")
    print("Aqui você pode carregar e interagir com seus PDFs.")
    print("Use o menu para navegar entre as opções disponíveis.")
    print("Divirta-se explorando seus documentos!")
    input("Pressione Enter para continuar...")
    os.system('cls' if os.name == 'nt' else 'clear')
    return None


def options_view() -> bool:
    print("Opções disponíveis:")
    print("1. Carregar PDF")
    print("2. Listar PDFs carregados")
    print("3. Interagir com um PDF")
    print("0. Sair")
    choice = input("Escolha uma opção: ")
    os.system('cls' if os.name == 'nt' else 'clear')

    match choice:
        case '0':
            print("Saindo do ChatPDF. Até a próxima!")
            controller.clear_session()
            return False
        case '1':
            pdf_path = input("Digite o caminho do arquivo PDF: ")
            if controller.load_pdf(pdf_path):
                print(f"PDF '{pdf_path}' carregado com sucesso!")
            else:
                print(f"Falha ao carregar o PDF {pdf_path}")
                print("Verifique se o caminho está correto e tente novamente.")
            input("Pressione Enter para continuar...")
            os.system('cls' if os.name == 'nt' else 'clear')
            return True
        case '2':
            pdfs = controller.list_pdf()
            if pdfs:
                print("PDFs carregados:")
                for pdf in pdfs:
                    print(f"- {pdf}")
            else:
                print("Nenhum PDF carregado.")
            input("Pressione Enter para continuar...")
            os.system('cls' if os.name == 'nt' else 'clear')
            return True
        case '3':
            message = input("Digite sua pergunta sobre os PDFs carregados: ")
            response = controller.chat(message)
            print(f"Resposta: {response}")
            input("Pressione Enter para continuar...")
            os.system('cls' if os.name == 'nt' else 'clear')
            return True
        case _:
            print("Opção inválida. Tente novamente.")
            input("Pressione Enter para continuar...")
            os.system('cls' if os.name == 'nt' else 'clear')
            return True
